"""CPU checks for provided code; diffusion and CFG exercises remain incomplete."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from models import UNet
from pipelines import DDPMPipeline
from schedulers import DDPMScheduler
from utils import AverageMeter, randn_tensor, seed_everything, str2bool
from utils.checkpoint import load_checkpoint, manage_checkpoints, save_checkpoint
from utils.dist import init_distributed_device_so


class BufferScheduler(nn.Module):
    """Checkpoint-interface test double, with no diffusion implementation."""

    def __init__(self, steps, persistent_timesteps=False):
        super().__init__()
        self.register_buffer("schedule", torch.tensor([0.25, 0.75]))
        self.register_buffer("timesteps", torch.arange(steps), persistent=persistent_timesteps)


class ProvidedComponentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_unet_forward_backward_unconditional_and_conditional(self):
        for conditional in (False, True):
            with self.subTest(conditional=conditional):
                model = UNet(
                    input_size=8, input_ch=3, T=10, ch=32, ch_mult=[1, 2],
                    attn=[1], num_res_blocks=1, conditional=conditional,
                    c_dim=16 if conditional else None,
                )
                images = torch.randn(2, 3, 8, 8)
                labels = torch.randn(2, 16) if conditional else None
                output = model(images, torch.tensor([1, 2]), c=labels)
                self.assertEqual(output.shape, images.shape)
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertIsNotNone(model.stem.weight.grad)
                self.assertTrue(torch.isfinite(model.stem.weight.grad).all())
                self.assertGreater(model.stem.weight.grad.abs().sum().item(), 0)
                if conditional:
                    self.assertIsNotNone(model.middleblocks[0].cross_attn.proj_v.weight.grad)
                with torch.no_grad():
                    self.assertEqual(model(images, 1, c=labels).shape, images.shape)

    def test_provided_cpu_utilities(self):
        seed_everything(123)
        first = randn_tensor((2, 3), device=torch.device("cpu"))
        seed_everything(123)
        self.assertTrue(torch.equal(first, randn_tensor((2, 3))))
        generators = [torch.Generator().manual_seed(seed) for seed in (7, 9)]
        batch = randn_tensor((2, 3), generator=generators, device=torch.device("cpu"))
        for index, seed in enumerate((7, 9)):
            expected = torch.randn((1, 3), generator=torch.Generator().manual_seed(seed))
            self.assertTrue(torch.equal(batch[index:index + 1], expected))
        meter = AverageMeter()
        meter.update(2, n=2)
        meter.update(5)
        self.assertEqual(meter.avg, 3)
        self.assertTrue(str2bool("true"))
        self.assertFalse(str2bool("false"))
        with mock.patch.dict("os.environ", {}, clear=True):
            state = init_distributed_device_so("cpu")
        self.assertEqual(state["device"], "cpu")
        self.assertFalse(state["distributed"])

    def test_conditioning_input_normalization_and_guidance_modes(self):
        pipeline = DDPMPipeline(nn.Linear(2, 2), BufferScheduler(2), class_embedder=nn.Identity())
        classes, enabled = pipeline._prepare_conditioning_inputs(2, None, None, "cpu")
        self.assertIsNone(classes)
        self.assertFalse(enabled)
        for scale, expected in ((None, False), (1.0, False), (2.0, True), (0.0, True)):
            with self.subTest(scale=scale):
                classes, enabled = pipeline._prepare_conditioning_inputs(2, 3, scale, "cpu")
                self.assertEqual(classes.tolist(), [3, 3])
                self.assertEqual(classes.dtype, torch.long)
                self.assertEqual(enabled, expected)
        classes, _ = pipeline._prepare_conditioning_inputs(2, [1, 2], 2.0, "cpu")
        self.assertEqual(classes.tolist(), [1, 2])
        with self.assertRaisesRegex(ValueError, "one class"):
            pipeline._prepare_conditioning_inputs(2, [1], 2.0, "cpu")
        with self.assertRaisesRegex(ValueError, "Provide classes"):
            pipeline._prepare_conditioning_inputs(2, None, 2.0, "cpu")
        with self.assertRaisesRegex(ValueError, "class_embedder"):
            DDPMPipeline(nn.Linear(2, 2), BufferScheduler(2))._prepare_conditioning_inputs(
                2, [1, 2], 2.0, "cpu"
            )

    def test_randn_accepts_list_shape_and_string_device_with_generators(self):
        generators = [torch.Generator().manual_seed(seed) for seed in (7, 9)]
        batch = randn_tensor([2, 3], generator=generators, device="cpu")
        self.assertEqual(tuple(batch.shape), (2, 3))
        for index, seed in enumerate((7, 9)):
            expected = torch.randn((1, 3), generator=torch.Generator().manual_seed(seed))
            self.assertTrue(torch.equal(batch[index:index + 1], expected))
        sampled = randn_tensor((2, 3), generator=torch.Generator().manual_seed(5), device="cpu")
        expected = torch.randn((2, 3), generator=torch.Generator().manual_seed(5))
        self.assertTrue(torch.equal(sampled, expected))

    def test_pipeline_image_conversion(self):
        pipeline = DDPMPipeline(nn.Linear(2, 2), BufferScheduler(2))
        pictures = pipeline.numpy_to_pil(np.full((2, 4, 4, 3), 0.5))
        self.assertEqual(len(pictures), 2)
        self.assertEqual(pictures[0].size, (4, 4))
        self.assertEqual(pictures[0].getpixel((0, 0)), (128, 128, 128))

    def test_checkpoint_round_trip_restores_models_optimizer_and_epoch(self):
        unet, vae, embedder = nn.Linear(3, 2), nn.Linear(2, 2), nn.Embedding(4, 2)
        scheduler = BufferScheduler(10)
        optimizer = torch.optim.Adam(unet.parameters(), lr=0.02)
        unet(torch.ones(1, 3)).square().sum().backward()
        optimizer.step()
        saved_weights = {key: value.clone() for key, value in unet.state_dict().items()}
        target, target_vae, target_embedder = nn.Linear(3, 2), nn.Linear(2, 2), nn.Embedding(4, 2)
        target_scheduler = BufferScheduler(3)
        target_optimizer = torch.optim.Adam(target.parameters(), lr=0.5)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            save_checkpoint(unet, scheduler, vae, embedder, optimizer, 4, directory)
            with mock.patch("utils.checkpoint.torch.load", wraps=torch.load) as loader:
                epoch = load_checkpoint(
                    target, target_scheduler, target_vae, target_embedder,
                    target_optimizer, str(Path(directory) / "checkpoint_epoch_4.pth"),
                )
            self.assertEqual(loader.call_args.kwargs["map_location"], "cpu")
            self.assertTrue(loader.call_args.kwargs["weights_only"])
        self.assertEqual(epoch, 4)
        for key, value in target.state_dict().items():
            self.assertTrue(torch.equal(value, saved_weights[key]))
        self.assertTrue(torch.equal(target_vae.weight, vae.weight))
        self.assertTrue(torch.equal(target_embedder.weight, embedder.weight))
        self.assertEqual(target_scheduler.timesteps.numel(), 3)
        self.assertEqual(target_optimizer.param_groups[0]["lr"], 0.02)
        self.assertTrue(target_optimizer.state)
        for source_state, target_state in zip(optimizer.state.values(), target_optimizer.state.values()):
            self.assertTrue(torch.equal(source_state["exp_avg"], target_state["exp_avg"]))

    def test_legacy_checkpoint_keeps_new_inference_timetable(self):
        source, target = nn.Linear(2, 2), nn.Linear(2, 2)
        legacy_scheduler = BufferScheduler(10, persistent_timesteps=True)
        target_scheduler = BufferScheduler(3, persistent_timesteps=True)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            save_checkpoint(source, legacy_scheduler, epoch=1, save_dir=directory)
            load_checkpoint(target, target_scheduler, checkpoint_path=str(Path(directory) / "checkpoint_epoch_1.pth"))
        self.assertTrue(torch.equal(target_scheduler.timesteps, torch.arange(3)))
        self.assertTrue(torch.equal(target_scheduler.schedule, legacy_scheduler.schedule))
        # Exercise the real scheduler's buffer registration without filling its TODOs.
        scheduler = DDPMScheduler(num_train_timesteps=10)
        scheduler.timesteps = torch.arange(3)
        self.assertNotIn("timesteps", scheduler.state_dict())

    def test_checkpoint_default_epoch_and_retention(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            model, scheduler = nn.Linear(2, 2), BufferScheduler(3)
            save_checkpoint(model, scheduler, save_dir=directory)
            self.assertTrue((Path(directory) / "checkpoint.pth").is_file())
            for epoch in range(12):
                save_checkpoint(model, scheduler, epoch=epoch, save_dir=directory)
            expected = {f"checkpoint_epoch_{epoch}.pth" for epoch in range(2, 12)}
            self.assertEqual({p.name for p in Path(directory).glob("checkpoint_epoch_*.pth")}, expected)
            legacy = Path(directory) / "checkpoint_epoch_None.pth"
            legacy.touch()
            manage_checkpoints(directory, keep_last_n=0)
            self.assertTrue(legacy.is_file())
            self.assertTrue((Path(directory) / "checkpoint.pth").is_file())
            self.assertFalse(any(Path(directory).glob("checkpoint_epoch_[0-9]*.pth")))


if __name__ == "__main__":
    unittest.main()
