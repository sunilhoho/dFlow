# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A Hydra-compatible training script for SiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import logging
import os
from pathlib import Path
from omegaconf import DictConfig

from dflow.models.sit import SiT
from dflow.transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
import wandb


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if not dist.is_initialized() or dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def train_sit_model(cfg: DictConfig):
    """
    Train a SiT model using Hydra configuration.
    """
    # Setup DDP if multiple GPUs available
    use_ddp = torch.cuda.device_count() > 1
    if use_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    device = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
    seed = cfg.seed * world_size + rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    
    # Setup experiment folder
    if rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        experiment_index = len(glob(f"{cfg.output_dir}/*"))
        model_string_name = cfg.model.name.replace("/", "-")
        experiment_name = f"{experiment_index:03d}-{model_string_name}-{cfg.transport.path_type}-{cfg.transport.prediction}"
        experiment_dir = f"{cfg.output_dir}/{experiment_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model
    assert cfg.model.input_size % 8 == 0, "Input size must be divisible by 8 (for the VAE encoder)."
    latent_size = cfg.model.input_size // 8
    
    model = SiT(
        input_size=latent_size,
        patch_size=cfg.model.patch_size,
        in_channels=cfg.model.in_channels,
        hidden_size=cfg.model.hidden_size,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        class_dropout_prob=cfg.model.class_dropout_prob,
        num_classes=cfg.model.num_classes,
        learn_sigma=cfg.model.learn_sigma
    )

    # Create EMA model
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    
    # Load checkpoint if provided
    if hasattr(cfg, 'checkpoint') and cfg.checkpoint is not None:
        ckpt_path = cfg.checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        ema.load_state_dict(checkpoint["ema_state_dict"])
        logger.info(f"Loaded checkpoint from {ckpt_path}")

    # Setup DDP
    if use_ddp:
        model = DDP(model.to(device), device_ids=[rank])
    
    # Create transport
    transport = create_transport(
        cfg.transport.path_type,
        cfg.transport.prediction,
        cfg.transport.loss_weight,
        cfg.transport.train_eps,
        cfg.transport.sample_eps
    )
    transport_sampler = Sampler(transport)
    
    # Create VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{cfg.vae}").to(device)
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

    # Setup data
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, cfg.model.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    dataset = ImageFolder(cfg.dataset.data_path, transform=transform)
    
    if use_ddp:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=cfg.seed
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.dataset.batch_size // world_size,
            shuffle=False,
            sampler=sampler,
            num_workers=cfg.dataset.num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=True,
            num_workers=cfg.dataset.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    logger.info(f"Dataset contains {len(dataset):,} images ({cfg.dataset.data_path})")

    # Prepare models for training
    update_ema(ema, model.module if use_ddp else model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # Labels to condition the model with
    local_batch_size = cfg.dataset.batch_size // world_size if use_ddp else cfg.dataset.batch_size
    ys = torch.randint(cfg.model.num_classes, size=(local_batch_size,), device=device)
    use_cfg = cfg.get('cfg_scale', 1.0) > 1.0
    
    # Create sampling noise
    n = ys.size(0)
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Setup classifier-free guidance
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([cfg.model.num_classes] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=cfg.cfg_scale)
        model_fn = ema.forward_with_cfg if hasattr(ema, 'forward_with_cfg') else ema.forward
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward

    logger.info(f"Training for {cfg.training.max_epochs} epochs...")
    
    for epoch in range(cfg.training.max_epochs):
        if use_ddp:
            sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            with torch.no_grad():
                # Map input images to latent space + normalize latents
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
            model_kwargs = dict(y=y, return_act=cfg.get('disp', False))
            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module if use_ddp else model)

            # Log loss values
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            if train_steps % cfg.training.log_every_n_steps == 0:
                # Measure training speed
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                
                # Reduce loss history over all processes
                if use_ddp:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / world_size
                else:
                    avg_loss = running_loss / log_steps
                
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                
                if wandb.run is not None:
                    wandb.log({
                        "train_loss": avg_loss,
                        "train_steps/sec": steps_per_sec,
                        "step": train_steps
                    })
                
                # Reset monitoring variables
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save SiT checkpoint
            if train_steps % cfg.training.save_every_n_steps == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model_state_dict": model.module.state_dict() if use_ddp else model.state_dict(),
                        "ema_state_dict": ema.state_dict(),
                        "opt_state_dict": opt.state_dict(),
                        "args": cfg
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                if use_ddp:
                    dist.barrier()
            
            # Generate samples
            if train_steps % cfg.training.sample_every_n_steps == 0 and train_steps > 0:
                logger.info("Generating EMA samples...")
                sample_fn = transport_sampler.sample_ode()  # default to ode sampling
                samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                
                if use_ddp:
                    dist.barrier()

                if use_cfg:  # remove null samples
                    samples, _ = samples.chunk(2, dim=0)
                samples = vae.decode(samples / 0.18215).sample
                
                if use_ddp:
                    out_samples = torch.zeros((cfg.dataset.batch_size, 3, cfg.model.input_size, cfg.model.input_size), device=device)
                    dist.all_gather_into_tensor(out_samples, samples)
                else:
                    out_samples = samples
                
                if wandb.run is not None:
                    wandb.log({"samples": [wandb.Image(img) for img in out_samples[:8]]}, step=train_steps)
                
                logger.info("Generating EMA samples done.")

    model.eval()  # important! This disables randomized embedding dropout
    logger.info("Done!")
    cleanup()


def sample_from_model(cfg: DictConfig):
    """
    Sample from a trained SiT model using Hydra configuration.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    assert cfg.model.input_size % 8 == 0, "Input size must be divisible by 8 (for the VAE encoder)."
    latent_size = cfg.model.input_size // 8
    
    model = SiT(
        input_size=latent_size,
        patch_size=cfg.model.patch_size,
        in_channels=cfg.model.in_channels,
        hidden_size=cfg.model.hidden_size,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        class_dropout_prob=cfg.model.class_dropout_prob,
        num_classes=cfg.model.num_classes,
        learn_sigma=cfg.model.learn_sigma
    )
    
    # Load checkpoint if provided
    if hasattr(cfg, 'checkpoint') and cfg.checkpoint is not None:
        checkpoint = torch.load(cfg.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["ema_state_dict"])
        print(f"Loaded checkpoint from {cfg.checkpoint}")
    
    model = model.to(device)
    model.eval()
    
    # Create transport
    transport = create_transport(
        cfg.transport.path_type,
        cfg.transport.prediction,
        cfg.transport.loss_weight,
        cfg.transport.train_eps,
        cfg.transport.sample_eps
    )
    transport_sampler = Sampler(transport)
    
    # Create VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{cfg.vae}").to(device)
    
    # Generate samples
    with torch.no_grad():
        # Create sampling noise
        n = cfg.sampling.num_samples
        zs = torch.randn(n, 4, latent_size, latent_size, device=device)
        ys = torch.randint(cfg.model.num_classes, size=(n,), device=device)
        
        # Setup classifier-free guidance
        use_cfg = cfg.get('cfg_scale', 1.0) > 1.0
        if use_cfg:
            zs = torch.cat([zs, zs], 0)
            y_null = torch.tensor([cfg.model.num_classes] * n, device=device)
            ys = torch.cat([ys, y_null], 0)
            sample_model_kwargs = dict(y=ys, cfg_scale=cfg.cfg_scale)
            model_fn = model.forward_with_cfg if hasattr(model, 'forward_with_cfg') else model.forward
        else:
            sample_model_kwargs = dict(y=ys)
            model_fn = model.forward
        
        # Sample
        sample_fn = transport_sampler.sample_ode()
        samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
        
        if use_cfg:  # remove null samples
            samples, _ = samples.chunk(2, dim=0)
        
        # Decode samples
        samples = vae.decode(samples / 0.18215).sample
        
        # Save samples
        output_dir = Path(cfg.output_dir) / "samples"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to images and save
        for i, sample in enumerate(samples):
            # Convert from [-1, 1] to [0, 1]
            sample = (sample + 1) / 2
            sample = torch.clamp(sample, 0, 1)
            
            # Convert to PIL Image
            sample_np = sample.permute(1, 2, 0).cpu().numpy()
            sample_np = (sample_np * 255).astype(np.uint8)
            sample_img = Image.fromarray(sample_np)
            
            # Save image
            sample_img.save(output_dir / f"sample_{i:04d}.png")
        
        print(f"Generated {len(samples)} samples and saved to {output_dir}")
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({"samples": [wandb.Image(img) for img in samples[:8]]})
