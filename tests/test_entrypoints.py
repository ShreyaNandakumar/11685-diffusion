"""Offline checks for the scaffold; these do not solve or grade the exercises."""

import ast
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import parse_args


class EntrypointTests(unittest.TestCase):
    def run_script(self, name, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / name), *args], cwd=ROOT,
            env={**os.environ, "WANDB_MODE": "disabled", "PYTHONDONTWRITEBYTECODE": "1"},
            text=True, capture_output=True, timeout=60,
        )

    def parse_config(self, text, *args):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "config.yaml"
            config.write_text(text)
            return parse_args(["--config", str(config), *args])

    def test_all_python_sources_parse(self):
        for path in ROOT.rglob("*.py"):
            with self.subTest(path=path.relative_to(ROOT)):
                ast.parse(path.read_text(), filename=str(path))

    def test_both_commands_show_help(self):
        for name in ("train.py", "inference.py"):
            with self.subTest(script=name):
                result = self.run_script(name, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--device", result.stdout)

    def test_unfinished_default_config_explains_todos(self):
        for name in ("train.py", "inference.py"):
            with self.subTest(script=name):
                result = self.run_script(name, "--config", "configs/ddpm.yaml")
                self.assertEqual(result.returncode, 2)
                self.assertIn("Complete the config TODOs", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_cli_overrides_null_config_values(self):
        args = self.parse_config("batch_size: null\nuse_cfg: true\nprediction_type: epsilon\n",
                                 "--batch_size", "2", "--device", "cpu")
        self.assertEqual(args.batch_size, 2)
        self.assertTrue(args.use_cfg)
        self.assertEqual(args.prediction_type, "epsilon")
        self.assertEqual(args.device, "cpu")

    def test_unknown_config_key_is_rejected(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            self.parse_config("predictor_type: epsilon\n")
        self.assertIn("Unknown config keys: predictor_type", stderr.getvalue())

    def test_training_config_snapshot_can_be_reloaded(self):
        import ruamel.yaml
        args = self.parse_config("{}\n", "--batch_size", "3", "--device", "cpu")
        # These fields are added during training before vars(args) is saved.
        args.distributed = True
        args.world_size = 4
        args.rank = 2
        args.local_rank = 2
        args.total_batch_size = 12
        args.max_train_steps = 100
        args.epoch = 3
        stream = io.StringIO()
        ruamel.yaml.YAML().dump(vars(args), stream)
        loaded = self.parse_config(stream.getvalue())
        self.assertEqual(loaded.batch_size, 3)
        self.assertEqual(loaded.device, "cpu")
        self.assertFalse(hasattr(loaded, 'rank'))
        self.assertFalse(hasattr(loaded, 'world_size'))

    def test_bad_config_values_are_rejected(self):
        for config in ("- not-a-mapping\n", "batch_size: 0\n", "num_workers: -1\n",
                       "num_train_timesteps: 10\nnum_inference_steps: 20\n",
                       "beta_start: 0.5\nbeta_end: 0.1\n"):
            with self.subTest(config=config):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.parse_config(config)

    def test_cpu_training_reaches_explicit_exercise_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "config.yaml"
            config.write_text("{}\n")
            result = self.run_script("train.py", "--config", str(config), "--device", "cpu")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Complete the dataset and dataloader TODOs", result.stderr)
        self.assertNotIn("CUDA is not available", result.stderr)

    def test_inference_reaches_explicit_exercise_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "config.yaml"
            config.write_text("{}\n")
            checkpoint = Path(folder) / "not_loaded_yet.pth"
            checkpoint.write_bytes(b"The unfinished scheduler stops before checkpoint loading.")
            result = self.run_script("inference.py", "--config", str(config), "--device", "cpu",
                                     "--ckpt", str(checkpoint))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Complete the scheduler setup TODO", result.stderr)
        self.assertNotIn("UnboundLocalError", result.stderr)

    def test_missing_inference_checkpoint_is_explained(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "config.yaml"
            config.write_text("{}\n")
            result = self.run_script("inference.py", "--config", str(config), "--device", "cpu")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Provide a trained checkpoint", result.stderr)

    def test_metric_imports_need_no_weight_download(self):
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.inception import InceptionScore
        self.assertTrue(callable(FrechetInceptionDistance))
        self.assertTrue(callable(InceptionScore))


if __name__ == "__main__":
    unittest.main()
