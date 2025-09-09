from __future__ import annotations

import logging
from pathlib import Path

import plotly.graph_objects as go
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from PIL.Image import Image
import torch.distributed as dist
import wandb
import hashlib
import os
import math

def is_main_process():
    return dist.get_rank() == 0

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }

def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)

def initialize(args, entity, exp_name, project_name):
    config_dict = namespace_to_dict(args)
    wandb.login(key=os.environ["WANDB_KEY"])
    wandb.init(
        entity=entity,
        project=project_name,
        name=exp_name,
        config=config_dict,
        id=generate_run_id(exp_name),
        resume="allow",
    )


def format_fig(
    fig: Image | go.Figure | plt.Figure,
) -> go.Figure | plt.Figure | wandb.Image:
    """
    Convert a matplotlib figure/plotly figure/PIL image to a wandb image
    """
    if isinstance(fig, (Image, plt.Figure)):
        return wandb.Image(fig)
    return fig


def check_wandb(fun):
    def inner(*args, **kwargs):
        if (
            isinstance(wandb.run, wandb.sdk.wandb_run.Run)
            and wandb.run.settings.mode == "run"
        ):
            return fun(*args, **kwargs)
        elif wandb.run is None:
            mode = "none"
        elif isinstance(wandb.run.mode, wandb.sdk.lib.disabled.RunDisabled):
            mode = "disabled"
        else:
            mode = wandb.run.settings.mode
        logging.warning(
            "Wandb not available (mode=%s): Unable to call function %s.",
            mode,
            fun.__name__,
        )

    return inner

def log(stats, step=None):
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)

def log_image(sample, step=None):
    if is_main_process():
        sample = array2grid(sample)
        wandb.log({f"samples": wandb.Image(sample), "train_step": step})

def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(-1,1))
    x = x.mul(255).add_(0.5).clamp_(0,255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x

@check_wandb
def merge_wandb_cfg(cfg: DictConfig | dict) -> DictConfig:
    wandb_config = dict(wandb.run.config)
    wandb_config.pop("hydra", None)
    cfg = OmegaConf.merge(wandb_config, cfg)
    logging.info("Merged config with wandb config.")
    return cfg


@check_wandb
def upload_ckpt(path: Path | str, name: str = "ckpt"):
    name = f"ckpt/{name}"
    model_artifact = wandb.Artifact(
        wandb.run.id, type="model", metadata={"path": str(path), "name": name}
    )
    model_artifact.add_file(str(path), name=name)
    wandb.log_artifact(model_artifact)
    logging.info("Uploaded checkpoint %s to wandb.", name)


@check_wandb
def restore_ckpt(out_dir: Path | str):
    try:
        artifact = wandb.run.use_artifact(f"{wandb.run.id}:latest")
        ckpt = artifact.download(out_dir)
        logging.info(
            "Checkpoint %s restored from wandb.",
            artifact.metadata.get("name", ckpt),
        )
    except wandb.CommError as exception:
        logging.debug("Wandb raised exception %s", exception)
        logging.info("No previous checkpoints found for wandb id %s.", wandb.run.id)


@check_wandb
def delete_old_wandb_ckpts():
    try:
        run = wandb.Api().run(wandb.run.path)
        for artifact in run.logged_artifacts():
            if len(artifact.aliases) == 0:
                # Clean up versions that don't have an alias such as 'latest'
                artifact.delete()
                logging.info(
                    "Marked checkpoint %s for deletion on wandb.",
                    artifact.metadata["name"],
                )
    except wandb.CommError as exception:
        logging.debug("Wandb raised exception %s", exception)
        logging.warning("Unable to delete checkpoints on wandb.")
