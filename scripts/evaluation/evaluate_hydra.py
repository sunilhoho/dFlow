"""
DDP-capable evaluation script for dFlow (naive + EMA).
- Each rank generates a disjoint chunk of samples.
- Rank 0 collects, loads real data and computes metrics.
- tqdm used for progress reporting.
"""

import os
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# metric imports (torchmetrics >=0.11 style)
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from diffusers.models import AutoencoderKL

from dflow.transport import Sampler
from dflow.utils import hydra as hydra_utils


def setup_ddp(cfg):
    use_ddp = cfg.get("use_ddp", False) or cfg.get("training", {}).get("use_ddp", False)
    initialized = False
    if use_ddp:
        dist.init_process_group(backend="nccl")
        initialized = True
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(0)
    return use_ddp, initialized, rank, world_size, device


def balanced_local_count(total, world_size, rank):
    base = total // world_size
    rem = total % world_size
    return base + (1 if rank < rem else 0)


@hydra.main(version_base=None, config_path="../../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    print("Evaluation Configuration:")
    print(OmegaConf.to_yaml(cfg))

    use_ddp, ddp_inited, rank, world_size, device = setup_ddp(cfg)
    print(f"[rank {rank}] Using device: {device} (world_size={world_size}, use_ddp={use_ddp})")

    # Load checkpoint (every rank loads)
    ckpt_path = os.path.join(cfg.output_dir, "train", "checkpoints", "final.pt")
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint not found at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    if rank == 0:
        print(f"[rank 0] Loaded checkpoint from {ckpt_path}")
        if isinstance(checkpoint, dict):
            print(f"[rank 0] checkpoint keys: {list(checkpoint.keys())[:10]}")

    def extract_state_dict(checkpoint, key_candidates):
        """Extract a state dict from checkpoint (handles dict or raw state_dict)."""
        if isinstance(checkpoint, dict):
            for key in key_candidates:
                if key in checkpoint:
                    return checkpoint[key]
        # If checkpoint itself looks like a state_dict
        if all(isinstance(k, str) for k in checkpoint.keys()):
            return checkpoint
        return None

    model_variants = {
        "naive": extract_state_dict(checkpoint, ["model_state_dict", "model"]),
        "ema": extract_state_dict(checkpoint, ["ema_state_dict", "ema"]),
    }

    num_eval_samples = int(cfg.get("evaluation", {}).get("num_eval_samples", 50000))
    local_target = balanced_local_count(num_eval_samples, world_size, rank)

    # Gather per-rank sizes so rank 0 can reconstruct results
    local_target_tensor = torch.tensor([local_target], device=device)
    if use_ddp:
        gathered_sizes = [torch.zeros_like(local_target_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_sizes, local_target_tensor)
        sizes = [int(x.item()) for x in gathered_sizes]
    else:
        sizes = [local_target]

    max_local = max(sizes)
    results_summary = {}

    for variant_name, state_dict in model_variants.items():
        if state_dict is None:
            if rank == 0:
                print(f"[WARN] No state dict found for variant '{variant_name}', skipping.")
            continue

        if rank == 0:
            print(f"\n=== Evaluating {variant_name.upper()} model ===")

        # instantiate model and load weights (per-rank)
        model = hydra.utils.instantiate(cfg.model).to(device)

        # Try to load normally
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
        except RuntimeError:
            # Retry after stripping "module." prefix
            new_state = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state[k[len("module.") :]] = v
                else:
                    new_state[k] = v
            missing, unexpected = model.load_state_dict(new_state, strict=False)

        if rank == 0:
            total_params = sum(p.numel() for p in model.state_dict().values())
            loaded_params = total_params - sum(
                model.state_dict()[k].numel() for k in missing if k in model.state_dict()
            )
            print(f"[rank 0] Weight loading summary for {variant_name}:")
            print(f"  Total params:      {total_params}")
            print(f"  Loaded params:     {loaded_params}")
            print(f"  Missing keys:      {len(missing)}")
            if missing:
                print(f"    {missing}")
            print(f"  Unexpected keys:   {len(unexpected)}")
            if unexpected:
                print(f"    {unexpected}")

        model.eval()


        transport = hydra.utils.instantiate(cfg.transport)
        transport_sampler = Sampler(transport)
        if cfg.dataset.image_size == 256:
            vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
        # per-variant outdir (only by rank 0)
        variant_outdir = Path(cfg.output_dir) / "evaluation" / variant_name
        if rank == 0:
            variant_outdir.mkdir(parents=True, exist_ok=True)

        # generation params
        gen_batch_size = int(cfg.get("evaluation", {}).get("gen_batch_size", cfg.dataset.loader.batch_size))
        gen_batch_size = max(1, gen_batch_size)
        gen_batch_size = min(gen_batch_size, max_local)

        use_cfg = cfg.get("cfg_scale", 1.0) > 1.0

        # Choose sampling function once per variant
        with torch.no_grad():
            if cfg.sampling.method == "ode":
                sample_fn = transport_sampler.sample_ode(sampling_method=cfg.sampling.solver, num_steps=cfg.sampling.num_steps+1)
            elif cfg.sampling.method == "sde":
                sample_fn = transport_sampler.sample_sde(sampling_method=cfg.sampling.solver, num_steps=cfg.sampling.num_steps+1)
            else:
                raise ValueError(f"Unknown sampling method: {cfg.sampling.method}")

            local_chunks = []
            produced = 0
            pbar = tqdm(total=local_target, desc=f"Rank {rank} gen {variant_name}", disable=False)

            torch.manual_seed(cfg.seed + rank)
            while produced < local_target:
                cur = min(gen_batch_size, local_target - produced)
                ys = torch.randint(cfg.model.num_classes, size=(cur,), device=device)
                if cfg.dataset.image_size == 256:
                    zs = torch.randn(cur, 4, cfg.model.input_size, cfg.model.input_size, device=device)
                else:
                    zs = torch.randn(cur, 3, cfg.model.input_size, cfg.model.input_size, device=device)

                if use_cfg:
                    ys_cat = torch.cat([ys, torch.tensor([cfg.model.num_classes] * cur, device=device)], dim=0)
                    zs_cat = torch.cat([zs, zs], dim=0)
                    batch = sample_fn(zs_cat, model.forward_with_cfg, y=ys_cat, cfg_scale=cfg.cfg_scale)[-1]
                    batch, _ = batch.chunk(2, dim=0)
                else:
                    batch = sample_fn(zs, model.forward, y=ys)[-1]
                if cfg.dataset.image_size == 256:
                    samples = vae.decode(samples / 0.18215).sample
                    samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1)
                else:
                    batch = torch.clamp((batch + 1) / 2, 0.0, 1.0)
                local_chunks.append(batch)
                produced += cur
                pbar.update(cur)
            pbar.close()

        if len(local_chunks) == 0:
            local_samples = torch.zeros((0, 3, cfg.dataset.image_size, cfg.dataset.image_size), device=device)
        else:
            local_samples = torch.cat(local_chunks, dim=0)

        assert local_samples.shape[0] == local_target, f"[rank {rank}] produced {local_samples.shape[0]} != expected {local_target}"

        # pad to max_local
        b, c, h, w = local_samples.shape
        if b < max_local:
            pad = torch.zeros((max_local - b, c, h, w), device=device, dtype=local_samples.dtype)
            padded_local = torch.cat([local_samples, pad], dim=0)
        else:
            padded_local = local_samples

        # gather padded tensors across ranks
        gathered = [torch.zeros_like(padded_local) for _ in range(world_size)]
        if use_ddp:
            dist.all_gather(gathered, padded_local)
        else:
            gathered = [padded_local]

        # reconstruct on rank 0
        all_samples = None
        if rank == 0:
            reconstructed = []
            for r in range(world_size):
                s = sizes[r]
                if s == 0:
                    continue
                reconstructed.append(gathered[r][:s].cpu())
            if len(reconstructed) == 0:
                all_samples = torch.zeros((0, c, h, w), dtype=padded_local.dtype)
            else:
                all_samples = torch.cat(reconstructed, dim=0)
            all_samples = all_samples[:num_eval_samples]
            print(f"[rank 0] Reconstructed generated samples: {len(all_samples)}")

        # load real data on rank 0 only
        real_data = None
        if rank == 0:
            try:
                dataset_root = cfg.dataset.dataset.root
                print(f"[rank 0] Loading real data from {dataset_root}")

                try:
                    transform = hydra.utils.instantiate(cfg.dataset.transform)
                except Exception:
                    transform = transforms.Compose([
                        transforms.Resize(cfg.dataset.image_size),
                        transforms.CenterCrop(cfg.dataset.image_size),
                        transforms.ToTensor(),
                    ])

                # handle common dataset types
                root_lower = str(dataset_root).lower()
                if "cifar" in root_lower or cfg.dataset.get("name", "").lower().startswith("cifar"):
                    dataset = datasets.CIFAR10(root=dataset_root, train=True, download=False, transform=transform)
                else:
                    dataset = datasets.ImageFolder(root=dataset_root, transform=transform)

                dl = DataLoader(dataset, batch_size=int(cfg.dataset.loader.batch_size), shuffle=False,
                                num_workers=cfg.dataset.loader.num_workers if "loader" in cfg.dataset and "num_workers" in cfg.dataset.loader else 4,
                                pin_memory=True)
                real_chunks = []
                loaded = 0
                pbar_real = tqdm(total=num_eval_samples, desc="[rank 0] Loading real data")
                for x, _ in dl:
                    if loaded >= num_eval_samples:
                        break
                    take = min(x.shape[0], num_eval_samples - loaded)
                    real_chunks.append(x[:take])
                    loaded += take
                    pbar_real.update(take)
                pbar_real.close()

                if len(real_chunks) == 0:
                    real_data = None
                else:
                    real_data = torch.cat(real_chunks, dim=0)[:num_eval_samples].to(device)
                    print(f"[rank 0] Loaded {len(real_data)} real samples")
            except Exception as e:
                print(f"[rank 0] ⚠️ failed to load real data from {cfg.dataset.dataset.root}: {e}")
                real_data = None

        # --- Metrics computed on rank 0 only ---
        if rank == 0:
            samples = all_samples.to(device) if all_samples is not None else torch.zeros((0, c, h, w), device=device)
            metrics = {}

            # Helper: try to create metric with various safe kwargs
            def make_metric_safe(metric_cls, *args, device=device, **kwargs):
                """
                Try different ways to instantiate metric_cls so it won't perform DDP collectives.
                Returns (metric_instance, used_device)
                """
                # 1) try ddp=False
                try:
                    metric = metric_cls(*args, ddp=False, **kwargs).to(device)
                    return metric, device
                except TypeError:
                    pass
                except Exception:
                    pass

                # 2) try sync_on_compute=False
                try:
                    metric = metric_cls(*args, sync_on_compute=False, **kwargs).to(device)
                    return metric, device
                except TypeError:
                    pass
                except Exception:
                    pass

                # 3) instantiate normally then try to set attribute (if present)
                try:
                    metric = metric_cls(*args, **kwargs).to(device)
                    # try to set attribute to disable syncing
                    if hasattr(metric, "sync_on_compute"):
                        try:
                            setattr(metric, "sync_on_compute", False)
                        except Exception:
                            pass
                    return metric, device
                except Exception:
                    pass

                # 4) fallback: create on CPU (no DDP collectives possible)
                try:
                    metric = metric_cls(*args, **kwargs)  # on CPU
                    return metric, torch.device("cpu")
                except Exception as e:
                    raise RuntimeError(f"Could not construct metric {metric_cls}: {e}")

            # ---------- FID ----------
            if real_data is not None:
                try:
                    fid, fid_device = make_metric_safe(FrechetInceptionDistance, feature=2048, normalize=True)
                    bs = min(int(cfg.dataset.loader.batch_size), 32)

                    # make sure data tensors are on the device where metric lives
                    real_dev = real_data.to(fid_device)
                    samples_dev = samples.to(fid_device)

                    # Update real data
                    for i in tqdm(range(0, len(real_dev), bs), desc="[rank 0] FID real"):
                        real_batch = real_dev[i:i+bs]
                        # many torchmetrics expect (N,H,W,C) uint8 or (N, C, H, W) depending on version.
                        # Your original code used uint8 HWC — keep that but adapt if your torchmetrics expects CHW.
                        fid.update((real_batch * 255).clamp(0, 255).to(torch.uint8), real=True)

                    # Update generated samples
                    for i in tqdm(range(0, len(samples_dev), bs), desc="[rank 0] FID fake"):
                        fake_batch = samples_dev[i:i+bs]
                        fid.update((fake_batch * 255).clamp(0, 255).to(torch.uint8), real=False)

                    # Compute (this should not block now)
                    metrics["num steps"] = int(cfg.sampling.num_steps)
                    metrics["fid"] = float(fid.compute())
                    print(f"[rank 0] FID: {metrics['fid']:.4f}")
                    results_summary[variant_name] = metrics
                except Exception as e:
                    print(f"[rank 0] ⚠️ Could not compute FID: {e}")

            # ---------- Inception Score ----------
            try:
                inception, inc_device = make_metric_safe(InceptionScore)
                bs = min(int(cfg.dataset.loader.batch_size), 32)

                samples_dev = samples.to(inc_device)

                for i in tqdm(range(0, len(samples_dev), bs), desc="[rank 0] Inception"):
                    batch = samples_dev[i:i+bs]
                    batch_uint8 = (batch * 255).clamp(0,255).to(torch.uint8)
                    inception.update(batch_uint8)

                is_mean, is_std = inception.compute()
                metrics["num steps"] = int(cfg.sampling.num_steps)
                metrics["inception_score_mean"] = float(is_mean)
                metrics["inception_score_std"] = float(is_std)
                print(f"[rank 0] Inception Score: {is_mean:.4f} ± {is_std:.4f}")
                results_summary[variant_name] = metrics

            except Exception as e:
                print(f"[rank 0] ⚠️ Could not compute Inception Score: {e}")

            # Save metrics
            metrics_path = os.path.join(variant_outdir, f"metrics_numsteps{cfg.sampling.num_steps}.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to: {metrics_path}")

        # sync before next variant
        if use_ddp:
            dist.barrier()

    # final summary write by rank 0
    if rank == 0:
        summary_path = Path(cfg.output_dir) / "evaluation" / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        # load old results if exists
        if summary_path.exists():
            with open(summary_path, "r") as f:
                all_results = json.load(f)
        else:
            all_results = {}
        step_key = f"steps_{cfg.sampling.num_steps}"
        if step_key not in all_results:
            all_results[step_key] = {}
        all_results[step_key].update(results_summary)

        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[rank 0] Summary saved to {summary_path}")
        print("✅ Evaluation completed.")

    if ddp_inited:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
