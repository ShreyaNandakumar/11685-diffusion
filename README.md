# IDL Fall 2026 - Guided Project: Diffusion

**Midterm report:** October 23, 2026  
**Final report:** TBD

There is no Kaggle competition or CSV submission this semester. Evaluate locally and include FID and IS results in your report.

# Setup and scaffold checks

Use Python 3.12 and create an isolated environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The tests use small synthetic inputs on CPU. They need no dataset, pretrained weights, GPU, or W&B login. They check the provided code and command-line setup; they do not implement or grade DDPM, DDIM, VAE, CFG, or FID/IS exercises.

This is an unfinished student scaffold. Fill the blank entries in `configs/ddpm.yaml` and complete the code TODOs before running training or inference. The untouched config intentionally stops with a list of missing values. For setup help, both `python train.py --help` and `python inference.py --help` should run.

# Starter Code Usage

The commands below apply **after completing the relevant TODOs**. CUDA is selected when available; use `--device cpu` for small debugging runs. Full training is intended for a GPU. Set `WANDB_MODE=disabled` to run without a W&B account.

**Training**

```
python train.py --config configs/ddpm.yaml
```

**Inference and Evaluating**

```
python inference.py --config configs/ddpm.yaml --ckpt /path/to/checkpoint
```

# 1. Download the data

Please first download the data from here: https://drive.google.com/drive/u/0/folders/1Hr8LU7HHPEad8ALmMo5cvisazsm6zE8Z

Create the data directory and extract the archive with

```
mkdir -p data
tar -xvf imagenet100_128x128.tar.gz -C data
```

# 2. Implementing DDPM from Scratch

This homework will start from implementing DDPM from scratch.

We provide the basic code structure for you and you will be implementing the following modules (by filling all TODOs):

```
1. pipelines/ddpm.py
2. schedulers/scheduling_ddpm.py
3. train.py
4. configs/ddpm.yaml
```

A very basic U-Net architecture is provided to you, and you will need to improve the architecture for better performance.

# 3. Implementing DDIM

Implement the DDIM from scratch:

```
1. schedulers/scheduling_ddim.py
2. create a config with ddim by setting use_ddim to True
```

**NOTE: you need to set use_ddim to TRUE**

# 4. Implementing Latent DDPM

Implement the Latent DDPM.

The VAE modules are provided. The pretrained VAE checkpoint is an external prerequisite.

[Download the pretrained VAE checkpoint](https://drive.google.com/file/d/1KSdtgHVOADazaZE2XflA61zw97iTKiRe/view) and save it as `pretrained/model.ckpt` (create the `pretrained` directory if needed). The checkpoint is not included in this package. Latent DDPM training and inference require it.

If the download is named `Model.ckpt`, rename it to lowercase `model.ckpt` when placing it in `pretrained/`.

You need to implement:

```
1. models/vae.py
2. train.py with vae related stuff
3. pipelines/ddpm.py with vae related stuff
```

**NOTE: you need to set latent_ddpm to TRUE**

# 5. Implementing CFG

Implement CFG

```
1. models/class_embedder.py
2. train.py with cfg related stuff
3. pipelines/ddpm.py with cfg related stuff
```

**NOTE: you need to set use_cfg to TRUE**

# 6. Evaluation

Complete the evaluation TODOs in `inference.py`, then run:

```bash
python inference.py --config configs/ddpm.yaml --ckpt /path/to/checkpoint
```

Generate 5,000 images. With classifier-free guidance, generate 50 images for each of the 100 classes. Without classifier-free guidance, generate 5,000 unconditional images.

Use `torchmetrics` to compute FID against the validation images and IS on the generated images. Include results for DDPM, DDPM + DDIM, latent DDPM, latent DDPM with CFG, and your advanced techniques in the report.

# 7. Deliverables

Submit your report and code, including local evaluation results. The midterm report is due **October 23, 2026** and covers DDPM and DDIM implementation and results. The final report is due **TBD** and covers all components and exploration. See the project writeup for the full requirements.
