"""
Main entry point for dFlow (Dispersive Flow) experiments.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import random
from time import sleep
import csv
import numpy as np

import hydra
import wandb
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import dflow.utils.hydra
import torch
from dflow.trainer.trainer import train_sit_model

logging.basicConfig(level=logging.INFO)

def setup_environment(config: DictConfig):
    """Setup environment and seeds."""
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    """Main function."""

    sleep(random.randint(1, 10))
    logging.info("---------------------------------------------------------------")
    
    # Log environment
    envs = {k: os.environ.get(k) for k in ["CUDA_VISIBLE_DEVICES", "PYTHONOPTIMIZE"]}
    logging.info("Env:\n%s", yaml.dump(envs))

    # Log overrides
    hydra_config = HydraConfig.get()
    logging.info("Command line args:\n%s", "\n".join(hydra_config.overrides.task))

    # Setup dir
    OmegaConf.set_struct(cfg, False)
    out_dir = Path(hydra_config.runtime.output_dir).absolute()
    logging.info("Hydra and wandb output path: %s", out_dir)
    if not cfg.get("output_dir"):
        cfg.output_dir = str(out_dir)
    logging.info("output path: %s", cfg.output_dir)
    cfg.model_name = cfg.model._target_.split(".")[-1]
    # Setup wandb
    tags = [cfg.model_name, cfg.dataset.name, ]
    if "wandb" not in cfg:
        cfg.wandb = OmegaConf.create()
    if not cfg.wandb.get("tags"):
        cfg.wandb.tags = tags
    if not cfg.wandb.get("group"):
        cfg.wandb.group = cfg.dataset.name

    if not cfg.wandb.get("id"):
        # create id based on log directory for automatic resuming
        sha = hashlib.sha256()
        sha.update(str(out_dir).encode())
        cfg.wandb.id = sha.hexdigest()

    if not cfg.wandb.get("name"):
        cfg.wandb.name = (
            f"{cfg.dataset.name}_{cfg.model_name}_sd{cfg.seed:03d}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    # Create save directory
    cfg.save_dir = f"{cfg.checkpoint_dir}/{cfg.dataset.name}_{cfg.model_name}_sd{cfg.seed:03d}/{cfg.wandb.name}"
    i = 1
    while os.path.exists(cfg.save_dir) and not cfg.get("override", False):
        cfg.save_dir += f"{i}/"
        i += 1

    OmegaConf.set_struct(cfg, True)
    wandb.init(
        dir=out_dir,
        **cfg.wandb,
    )

    # Resume old wandb run
    if wandb.run is not None and wandb.run.resumed:
        logging.info("Resume wandb run %s", wandb.run.path)

    # Log config and overrides
    logging.info("---------------------------------------------------------------")
    logging.info("Run config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    logging.info("---------------------------------------------------------------")

    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb_config["hydra"] = OmegaConf.to_container(hydra_config, resolve=True)

    for k in [
        "help",
        "hydra_help",
        "hydra_logging",
        "job_logging",
        "searchpath",
        "callbacks",
        "sweeper",
    ]:
        wandb_config["hydra"].pop(k, None)
    wandb.config.update(wandb_config, allow_val_change=True)

    # Run experiment
    logging.info("---------------------------------------------------------------")

    try:
        OmegaConf.resolve(cfg)
        
        # Setup environment
        setup_environment(cfg)
        # Run experiment based on job type
        # if cfg.job_type == "train":
        train_sit_model(cfg)
        # elif cfg.job_type == "sample":
        #     sample_from_model(cfg)
        # else:
            # logging.warning(f"Unknown job type: {cfg.job_type}")

        wandb.run.summary["error"] = None
        logging.info("Completed ✅")
        wandb.finish()

    except Exception as e:
        # Log error
        error_log_path = "logs/error_log.csv"
        fieldnames = ["timestamp", "model", "dataset", "loss", "error"]

        # Gather info for logging
        error_str = str(e)
        timestamp = datetime.now().isoformat()

        # Write to CSV (append, create header if file does not exist)
        try:
            write_header = False
            try:
                with open(error_log_path, "r", newline="") as f:
                    pass
            except FileNotFoundError:
                write_header = True

            with open(error_log_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "timestamp": timestamp,
                    "model": cfg.model.name,
                    "dataset": cfg.dataset.name,
                    "loss": cfg.loss.name,
                    "error": error_str
                })
        except Exception as log_exc:
            logging.error(f"Failed to log error to {error_log_path}: {log_exc}")

        logging.critical(e, exc_info=True)
        if wandb.run is not None:
            wandb.run.summary["error"] = str(e)
            wandb.finish(exit_code=1)


# def sync_wandb(wandb_dir: Path | str):
#     """Sync wandb runs."""
#     run_dirs = [f for f in Path(wandb_dir).iterdir() if "run-" in f.name]
#     for run_dir in sorted(run_dirs, key=os.path.getmtime):
#         logging.info("Syncing %s.", run_dir)
#         subprocess.run(
#             ["wandb", "sync", "--no-include-synced", "--mark-synced", str(run_dir)]
#         )


if __name__ == "__main__":
    main()
