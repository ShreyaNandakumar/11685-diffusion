"""Small CPU-only DDPM/DDIM experiment for midterm validation.

Does not change the assignment training pipeline. Uses CIFAR-10 through torchvision.
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Subset

from models import UNet
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--output-dir', default='./cpu_midterm_results')
    p.add_argument('--samples', type=int, default=128)
    p.add_argument('--steps', type=int, default=10, help='number of CPU training updates')
    p.add_argument('--image-size', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--sample-steps', type=int, default=10)
    p.add_argument('--download', action='store_true', help='download CIFAR-10 if needed')
    a = p.parse_args()
    if a.samples < 1 or a.steps < 1 or a.sample_steps < 1:
        p.error('samples, steps and sample-steps must be positive')
    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = datasets.CIFAR10(root=a.data_dir, train=True, download=a.download,
        transform=transforms.Compose([
            transforms.Resize((a.image_size, a.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ]))
    dataset = Subset(dataset, range(min(a.samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True, num_workers=0)
    model = UNet(input_size=a.image_size, input_ch=3, T=100, ch=32,
                 ch_mult=[1, 1], attn=[], num_res_blocks=1).to('cpu')
    scheduler = DDPMScheduler(num_train_timesteps=100, num_inference_steps=a.sample_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Show forward noising without training a model.
    images, _ = next(iter(loader))
    image = images[:1]
    noise = torch.randn_like(image)
    panels = [image]
    for t in [10, 30, 60, 99]:
        panels.append(scheduler.add_noise(image, noise, torch.tensor([t])))
    save_image((torch.cat(panels) + 1).clamp(0, 2) / 2, out/'forward_diffusion.png', nrow=5)

    losses = []
    batches = iter(loader)
    for step in range(1, a.steps+1):
        try:
            images, _ = next(batches)
        except StopIteration:
            batches = iter(loader)
            images, _ = next(batches)
        noise = torch.randn_like(images)
        t = torch.randint(0, 100, (images.shape[0],), dtype=torch.long)
        noisy = scheduler.add_noise(images, noise, t)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(noisy, t)
        loss = F.mse_loss(prediction, noise)
        if not torch.isfinite(loss):
            raise RuntimeError('Training loss is not finite')
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f'Train step {step}/{a.steps}: loss={losses[-1]:.6f}', flush=True)
    with (out/'training_loss.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'loss'])
        writer.writerows(enumerate(losses, start=1))
    torch.save({'unet': model.state_dict(), 'training_steps': a.steps}, out/'cpu_smoke_checkpoint.pt')

    # These are sanity-check samples, not evidence of high-quality generation.
    model.eval()
    for name, sampler in [('ddpm', scheduler),
                          ('ddim', DDIMScheduler(num_train_timesteps=100, num_inference_steps=a.sample_steps))]:
        sampler = sampler.to('cpu')
        generator = torch.Generator(device='cpu').manual_seed(42)
        start = time.perf_counter()
        generated = DDPMPipeline(model, sampler)(batch_size=1,
            num_inference_steps=a.sample_steps, generator=generator, device='cpu')
        elapsed = time.perf_counter()-start
        generated[0].save(out/f'{name}_sample.png')
        print(f'{name.upper()} {a.sample_steps} steps: {elapsed:.2f}s', flush=True)
    print(f'Finished. Results in {out.resolve()}')


if __name__ == '__main__':
    main()
