import os 
import sys 
import argparse
import time
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb 
import logging 
from logging import getLogger as get_logger
from tqdm import tqdm 
from PIL import Image
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.utils  import make_grid

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint

from train import parse_args

logger = get_logger(__name__)


def main():
    # parse arguments
    args = parse_args()
    
    # seed everything
    seed_everything(args.seed)
    device = torch.device(args.device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    
    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    if args.ckpt is None:
        raise ValueError("Provide a trained checkpoint with --ckpt /path/to/checkpoint.")
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    scheduler_class = DDIMScheduler if args.use_ddim else DDPMScheduler
    # TODO: instantiate scheduler_class using the configuration arguments.
    scheduler = scheduler_class(
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        prediction_type=args.prediction_type,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range,
    )
    if scheduler is None:
        raise NotImplementedError("Complete the scheduler setup TODO in inference.py before sampling.")
    scheduler = scheduler.to(device)
    
    # setup model
    logger.info("Creating model")
    # unet
    unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    # preint number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")
    
    # vae 
    vae = None
    if args.latent_ddpm:        
        vae = VAE()
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()
    # cfg
    class_embedder = None
    if args.use_cfg:
        # TODO: class embeder
        class_embedder = ClassEmbedder(None)
        
    # send to device
    unet = unet.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)
        
    # load checkpoint
    load_checkpoint(unet, scheduler, vae=vae, class_embedder=class_embedder, checkpoint_path=args.ckpt)
    unet.eval()
    if class_embedder is not None:
        class_embedder.eval()
    
    # TODO: pipeline
    pipeline = DDPMPipeline(
        unet,
        scheduler,
        vae=vae,
        class_embedder=class_embedder,
    )

    
    logger.info("***** Running Infrence *****")

    # TODO: using torchmetrics for evaluation, check the documents of torchmetrics
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    # Keep metric networks on CUDA when available. CPU is more broadly supported
    # for evaluation when sampling with MPS.
    metric_device = device if device.type == "cuda" else torch.device("cpu")
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(metric_device)
    inception = InceptionScore(normalize=False).to(metric_device)
    pil_to_tensor = transforms.PILToTensor()
    
    # TODO: we run inference to generation 5000 images
    # TODO: with cfg, we generate 50 images per class 
    num_generated = 0
    batch_size = args.batch_size
    sampling_start = time.perf_counter()
    if args.use_cfg:
        # generate 50 images per class
        for i in tqdm(range(args.num_classes)):
            logger.info(f"Generating 50 images for class {i}")
            batch_size = 50
            classes = torch.full((batch_size,), i, dtype=torch.long, device=device)
            gen_images = None 
            fake_images = torch.stack([pil_to_tensor(image) for image in gen_images])
            fid.update(fake_images.to(metric_device), real=False)
            inception.update(fake_images.to(metric_device))
            num_generated += len(gen_images)
    else:
        # generate 5000 images
        for start in tqdm(range(0, 5000, args.batch_size)):
            batch_size = min(args.batch_size, 5000 - start)
            gen_images = pipeline(
                batch_size=batch_size,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                device=device,
            )
            fake_images = torch.stack([pil_to_tensor(image) for image in gen_images])
            fid.update(fake_images.to(metric_device), real=False)
            inception.update(fake_images.to(metric_device))
            num_generated += len(gen_images)

    sampling_seconds = time.perf_counter() - sampling_start
    if num_generated != 5000:
        raise RuntimeError(f"Expected 5000 generated images, got {num_generated}.")
    
    # TODO: load validation images as reference batch
    validation_dir = os.path.join(
        os.path.dirname(os.path.normpath(args.data_dir)), "validation"
    )
    if not os.path.isdir(validation_dir):
        raise FileNotFoundError(f"Validation directory not found: {validation_dir}")
    validation_dataset = datasets.ImageFolder(
        validation_dir,
        transform=transforms.PILToTensor(),
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    for real_images, _ in tqdm(validation_loader, desc="Validation features"):
        fid.update(real_images.to(metric_device), real=True)
    
    # TODO: compute FID and IS
    fid_score = fid.compute().item()
    is_mean, is_std = inception.compute()
    logger.info(f"Generated images: {num_generated}")
    logger.info(f"Sampling time: {sampling_seconds:.2f} seconds")
    logger.info(f"Sampling throughput: {num_generated / sampling_seconds:.4f} images/second")
    logger.info(f"FID: {fid_score:.6f}")
    logger.info(f"IS: {is_mean.item():.6f} +/- {is_std.item():.6f}")
    
        
    


if __name__ == '__main__':
    main()
