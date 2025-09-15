# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A Hydra-compatible training script for SiT using PyTorch DDP.
"""
import torch
import pdb
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from copy import deepcopy
from time import time

import os
from omegaconf import DictConfig

from dflow.transport import Sampler
from dflow.utils.trainer_utils import create_logger, center_crop_arr, update_ema, requires_grad, cleanup
from dflow.utils import wandb_utils
from hydra.utils import instantiate
from omegaconf import OmegaConf
import wandb
#################################################################################
#                                  Training Loop                                #
#################################################################################

def train_sit_model(cfg: DictConfig):
    """
    Train a SiT model using Hydra configuration.
    """
    # Setup DDP if multiple GPUs available
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    use_ddp = cfg.training.get('use_ddp', False)

    if use_ddp:
        # ---- Fail fast & clearer erroring on collectives ----
        # os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        # os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
        # os.environ.setdefault("NCCL_DEBUG", "WARN")
        # Optional, if you have quirky PCIe/IB topology:
        # os.environ.setdefault("NCCL_P2P_DISABLE", "1")

        dist.init_process_group("nccl")

        # >>> Use environment-provided local rank for device selection <<<
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        world_size = dist.get_world_size()

        # Keep global seed different per rank if desired
        seed = cfg.training.global_seed * world_size + rank

        # Set device BEFORE any .to(device) / CUDA allocs
        torch.cuda.set_device(device)

        assert cfg.dataset.loader.batch_size % world_size == 0, \
            "Batch size must be divisible by world size."
        local_batch_size = cfg.dataset.loader.batch_size // world_size // cfg.model.avg_vf
        print(f"Starting rank={rank}, device={device}, world={world_size}, seed={seed}.")
    else:
        rank = 0
        world_size = 1
        device = 0
        seed = cfg.training.global_seed
        torch.cuda.set_device(device)
        local_batch_size = cfg.dataset.loader.batch_size // cfg.model.avg_vf
        print(f"Starting seed={seed}.")

    torch.manual_seed(seed)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_dir = f"{cfg.output_dir}/train"
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, use_ddp)
        logger.info(f"Experiment directory created at {experiment_dir}")
        wandb.init(
            dir=cfg.output_dir,
            **cfg.wandb,
        )

        # Resume old wandb run
        if wandb.run is not None and wandb.run.resumed:
            logger.info("Resume wandb run %s", wandb.run.path)

        # Log config and overrides
        logger.info("---------------------------------------------------------------")
        logger.info("Run config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
        logger.info("---------------------------------------------------------------")
        wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        # wandb_config["hydra"] = OmegaConf.to_container(cfg.hydra, resolve=True)

        # for k in [
        #     "help",
        #     "hydra_help",
        #     "hydra_logging",
        #     "job_logging",
        #     "searchpath",
        #     "callbacks",
        #     "sweeper",
        # ]:
        #     wandb_config["hydra"].pop(k, None)
        wandb.config.update(wandb_config, allow_val_change=True)

    else:
        logger = create_logger(None, use_ddp)

    # Create model
    assert cfg.dataset.image_size % 8 == 0, "Input size must be divisible by 8 (for the VAE encoder)."

    model = instantiate(cfg.model)
    # Create EMA model
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    # Load checkpoint if provided
    if hasattr(cfg.training, 'checkpoint') and cfg.training.checkpoint is not None:
        ckpt_path = cfg.training.checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        ema.load_state_dict(checkpoint["ema_state_dict"])
        logger.info(f"Loaded checkpoint from {ckpt_path}")

    # Setup DDP
    if use_ddp:
        model = DDP(
            model.to(device),
            device_ids=[rank],
        )
    else:
        model = model.to(device)
    # Create transport
    transport = instantiate(cfg.transport)
    transport_sampler = Sampler(transport)

    # Create VAE
    vae = instantiate(cfg.vae).to(device)
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer
    opt = instantiate(cfg.optimizer, params=model.parameters())

    # Setup data
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, cfg.dataset.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    # dataset = ImageFolder(cfg.dataset.data_path, transform=transform)
    dataset = instantiate(cfg.dataset.dataset, transform=transform)

    if use_ddp:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=True,
            seed=cfg.training.global_seed
        )
        loader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            sampler=sampler,
            num_workers=cfg.dataset.loader.num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            shuffle=cfg.dataset.loader.shuffle,
            num_workers=cfg.dataset.loader.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    logger.info(f"Dataset contains {len(dataset):,} images ({cfg.dataset.name})")

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
    ys = torch.randint(cfg.model.num_classes, size=(16,), device=device)
    use_cfg = cfg.get('cfg_scale', 1.0) > 1.0

    # Create sampling noise
    n = ys.size(0)
    if cfg.dataset.image_size == 256:
        zs = torch.randn(n, 4, cfg.model.input_size, cfg.model.input_size, device=device)
    else:
        zs = torch.randn(n, 3, cfg.model.input_size, cfg.model.input_size, device=device)

    # Setup classifier-free guidance
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([cfg.model.num_classes] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=cfg.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward

    logger.info(f"Training for {cfg.training.max_epochs} epochs...")

    for epoch in range(cfg.training.max_epochs):
        if use_ddp:
            loader.sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # only do this if x' spatial size is 256
            if cfg.dataset.image_size == 256:
                with torch.no_grad():
                    # Map input images to latent space + normalize latents
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            model_kwargs = dict(y=y, return_act=cfg.training.get('disp', False), 
                                additional_loss=cfg.training.get('additional_loss', False),
                                var_loss=cfg.training.get('var_loss', 0.0), 
                                cov_loss=cfg.training.get('cov_loss', 0.0))
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
                    wandb_utils.log({
                        "train_loss": avg_loss,
                        "train_steps/sec": steps_per_sec,
                        "train/step": train_steps
                    })

                # Reset monitoring variables
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save SiT checkpoint
            if train_steps % cfg.training.save_every_n_steps == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model_state_dict": model.module.state_dict() if use_ddp else model.state_dict(),  # type: ignore
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
                if cfg.dataset.image_size == 256:
                    samples = vae.decode(samples / 0.18215).sample

                if use_ddp:
                    out_samples = torch.zeros((64, 3, cfg.dataset.image_size, cfg.dataset.image_size), device=device)
                    dist.all_gather_into_tensor(out_samples, samples)
                else:
                    out_samples = samples
                
                if wandb.run is not None:
                    wandb_utils.log_image(out_samples, train_steps)

                logger.info("Generating EMA samples done.")

    # Save SiT checkpoint
    if rank == 0:
        checkpoint = {
            "model_state_dict": model.module.state_dict() if use_ddp else model.state_dict(),  # type: ignore
            "ema_state_dict": ema.state_dict(),
            "opt_state_dict": opt.state_dict(),
            "args": cfg
        }
        checkpoint_path = f"{checkpoint_dir}/final.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    if use_ddp:
        dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    logger.info("Done!")
    cleanup()


# def sample_from_model(cfg: DictConfig):
#     """
#     Sample from a trained SiT model using Hydra configuration.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Create model
#     model = instantiate(cfg.model)
    
#     # Load checkpoint if provided
#     if hasattr(cfg, 'checkpoint') and cfg.checkpoint is not None:
#         checkpoint = torch.load(cfg.checkpoint, map_location=device)
#         model.load_state_dict(checkpoint["ema_state_dict"])
#         print(f"Loaded checkpoint from {cfg.checkpoint}")
    
#     model = model.to(device)
#     model.eval()
    
#     # Create transport
#     transport = instantiate(cfg.transport)
#     transport_sampler = Sampler(transport)
    
#     # Create VAE
#     vae = instantiate(cfg.vae).to(device)
    
#     # Generate samples
#     with torch.no_grad():
#         # Create sampling noise
#         n = cfg.sampling.num_samples
#         zs = torch.randn(n, 4, cfg.model.input_size, cfg.model.input_size, device=device)
#         ys = torch.randint(cfg.model.num_classes, size=(n,), device=device)

#         # Setup classifier-free guidance
#         use_cfg = cfg.training.get('cfg_scale', 1.0) > 1.0
#         if use_cfg:
#             zs = torch.cat([zs, zs], 0)
#             y_null = torch.tensor([cfg.model.num_classes] * n, device=device)
#             ys = torch.cat([ys, y_null], 0)
#             sample_model_kwargs = dict(y=ys, cfg_scale=cfg.training.cfg_scale)
#             model_fn = model.forward_with_cfg
#         else:
#             sample_model_kwargs = dict(y=ys)
#             model_fn = model.forward

#         # Sample
#         sample_fn = transport_sampler.sample_ode()
#         samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]

#         if use_cfg:  # remove null samples
#             samples, _ = samples.chunk(2, dim=0)

#         # Decode samples
#         samples = vae.decode(samples / 0.18215).sample
        
#         # Save samples
#         output_dir = Path(cfg.output_dir) / "samples"
#         output_dir.mkdir(parents=True, exist_ok=True)
        
#         # Convert to images and save
#         for i, sample in enumerate(samples):
#             # Convert from [-1, 1] to [0, 1]
#             sample = (sample + 1) / 2
#             sample = torch.clamp(sample, 0, 1)
            
#             # Convert to PIL Image
#             sample_np = sample.permute(1, 2, 0).cpu().numpy()
#             sample_np = (sample_np * 255).astype(np.uint8)
#             sample_img = Image.fromarray(sample_np)
            
#             # Save image
#             sample_img.save(output_dir / f"sample_{i:04d}.png")

#         print(f"Generated {len(samples)} samples and saved to {output_dir}")

#         # Log to wandb if available
#         if wandb.run is not None:
#             log_image(samples, 0)
