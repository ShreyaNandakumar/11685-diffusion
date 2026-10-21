"""Small checkpoint regressions; no pretrained weights or Lightning required."""

import contextlib
import io
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
from torch import nn

from models import VAE


class VAECheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "model.ckpt"
        self.source = nn.Linear(3, 2)
        with torch.no_grad():
            self.source.weight.fill_(0.25)
            self.source.bias.fill_(0.75)
        self.target = nn.Linear(3, 2)
        self.safe_globals_before = torch.serialization.get_safe_globals()

    def tearDown(self):
        self.assertCountEqual(torch.serialization.get_safe_globals(), self.safe_globals_before)

    def save_legacy_metadata(self, include_unrelated_global=False):
        """Create the old class-key pickle shape, then discard the fake modules."""
        lightning = types.ModuleType("pytorch_lightning")
        callbacks = types.ModuleType("pytorch_lightning.callbacks")
        checkpoint_module = types.ModuleType("pytorch_lightning.callbacks.model_checkpoint")
        callback_class = type("ModelCheckpoint", (), {"__module__": checkpoint_module.__name__})
        checkpoint_module.ModelCheckpoint = callback_class
        callbacks.model_checkpoint = checkpoint_module
        lightning.callbacks = callbacks
        modules = {
            lightning.__name__: lightning,
            callbacks.__name__: callbacks,
            checkpoint_module.__name__: checkpoint_module,
        }
        checkpoint = {
            "state_dict": self.source.state_dict(),
            "callbacks": {callback_class: {"best_model_score": torch.tensor(1.5)}},
        }
        if include_unrelated_global:
            unrelated_module = types.ModuleType("unrelated_training_metadata")
            unrelated_class = type("UnsupportedMetadata", (), {"__module__": unrelated_module.__name__})
            unrelated_module.UnsupportedMetadata = unrelated_class
            modules[unrelated_module.__name__] = unrelated_module
            checkpoint["unrelated_metadata"] = unrelated_class
        with mock.patch.dict(sys.modules, modules):
            torch.save(checkpoint, self.path)
        self.assertIsNot(sys.modules.get(checkpoint_module.__name__), checkpoint_module)

    def assert_weights_restored(self, load_result):
        self.assertEqual(load_result.missing_keys, [])
        self.assertEqual(load_result.unexpected_keys, [])
        for name, value in self.target.state_dict().items():
            self.assertTrue(torch.equal(value, self.source.state_dict()[name]))

    def test_plain_tensor_checkpoint_restores_weights(self):
        torch.save({"state_dict": self.source.state_dict()}, self.path)
        with contextlib.redirect_stdout(io.StringIO()):
            result = VAE.init_from_ckpt(self.target, self.path)
        self.assert_weights_restored(result)

    def test_legacy_callback_key_loads_without_lightning_and_without_global_allowlist_leak(self):
        self.save_legacy_metadata()
        # Verify that this fixture needs the narrow compatibility mapping.
        with self.assertRaises(pickle.UnpicklingError):
            torch.load(self.path, map_location="cpu", weights_only=True)
        with contextlib.redirect_stdout(io.StringIO()):
            result = VAE.init_from_ckpt(self.target, self.path)
        self.assert_weights_restored(result)
        # A later unrelated load must not inherit the callback allowlist entry.
        with self.assertRaises(pickle.UnpicklingError):
            torch.load(self.path, map_location="cpu", weights_only=True)

    def test_unrelated_global_is_rejected_without_unrestricted_fallback(self):
        self.save_legacy_metadata(include_unrelated_global=True)
        original_weights = {name: value.clone() for name, value in self.target.state_dict().items()}
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(pickle.UnpicklingError) as raised:
            VAE.init_from_ckpt(self.target, self.path)
        self.assertIn("unrelated_training_metadata.UnsupportedMetadata", str(raised.exception))
        for name, value in self.target.state_dict().items():
            self.assertTrue(torch.equal(value, original_weights[name]))


if __name__ == "__main__":
    unittest.main()
