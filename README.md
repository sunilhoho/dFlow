# dFlow: Dispersive Flow
<img src="dispersiveloss.png" width="700" height="700">

**dFlow** is a PyTorch library for diffusion models with dispersive loss regularization. This repo contains the official PyTorch implementation of Dispersive Loss.

> [**Diffuse and Disperse: Image Generation with Representation Regularization**](https://arxiv.org/abs/2506.09027)<br>
> [Runqian Wang](https://raywang4.github.io/), [Kaiming He](https://people.csail.mit.edu/kaiming/)
> <br>MIT<br>

We propose Dispersive Loss, a simple plug-and-play regularizer that effectively improves diffusion-based generative models. 
Our loss function encourages internal representations to disperse in the hidden space, analogous to contrastive self-supervised learning, with the key distinction that it requires no positive sample pairs and therefore does not interfere with the sampling process used for regression.

## 🚀 Quick Start

### Installation

Install dFlow directly from the repository:

```bash
git clone https://github.com/raywang4/DispLoss.git
cd DispLoss
pip install -e .
```

Or install with development dependencies:

```bash
pip install -e ".[dev]"
```

### Key:

Make sure you run
```
export NCCL_P2P_DISABLE=1
```
### Basic Usage

```python
import dflow
from dflow.models import SiT
from dflow.configs import get_model_config
from dflow.training import train_with_dispersive_loss
from dflow.sampling import sample_ode

# Create a model
config = get_model_config('SiT-XL/2')
model = SiT(**config)

# Train with dispersive loss
train_config = get_training_config('imagenet_256')
loss_config = get_dispersive_loss_config('default')

# Sample images
images = sample_ode(model, shape=(4, 3, 256, 256), num_steps=100)
```

## 📚 Library Structure

```
dFlow/
├── dflow/                    # Main library package
│   ├── __init__.py          # Library exports
│   ├── models.py            # SiT model implementations
│   ├── transport.py         # Transport-based diffusion methods
│   ├── training.py          # Training utilities and dispersive loss
│   ├── sampling.py          # ODE/SDE sampling methods
│   └── configs.py           # Configuration management
├── configs/                 # Configuration files
│   └── models/              # Model configurations
│       └── sit_configs.py   # SiT model variants
├── transport/               # Transport module (original)
├── setup.py                 # Installation script
├── pyproject.toml          # Modern Python packaging
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🔧 Core Implementation

The core implementation of Dispersive Loss is highlighted below:

```python
def disp_loss(self, z): # Dispersive Loss implementation (InfoNCE-L2 variant)
  z = z.reshape((z.shape[0],-1)) # flatten
  diff = th.nn.functional.pdist(z).pow(2)/z.shape[1] # pairwise distance
  diff = th.concat((diff, diff, th.zeros(z.shape[0]).cuda()))  # match JAX implementation of full BxB matrix
  return th.log(th.exp(-diff).mean()) # calculate loss
```

## 🏃‍♂️ Training

### Command Line Training

To train with Dispersive Loss, simply add the `--disp` argument to the training script:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model SiT-XL/2 --data-path /path/to/imagenet/train --disp
```

### Programmatic Training

```python
from dflow.training import train_with_dispersive_loss, DispersiveLoss
from dflow.configs import get_model_config, get_training_config

# Get configurations
model_config = get_model_config('SiT-XL/2')
train_config = get_training_config('imagenet_256')

# Create model and loss
model = SiT(**model_config)
dispersive_loss = DispersiveLoss(lambda_disp=0.25, tau=1.0)

# Training loop
for epoch in range(train_config['num_epochs']):
    avg_loss = train_with_dispersive_loss(model, dataloader, optimizer, device, train_config)
    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
```

**Logging.** To enable `wandb`, firstly set `WANDB_KEY`, `ENTITY`, and `PROJECT` as environment variables:

```bash
export WANDB_KEY="key"
export ENTITY="entity name"
export PROJECT="project name"
```
Then in training command add the `--wandb` flag:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model SiT-XL/2 --data-path /path/to/imagenet/train --disp --wandb
```

**Resume training.** To resume training from custom checkpoint:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model SiT-L/2 --data-path /path/to/imagenet/train --disp --ckpt /path/to/model.pt
```

## 🎨 Sampling

### Command Line Sampling

**Pre-trained checkpoints.** We provide a [SiT-B/2 checkpoint](https://drive.google.com/file/d/18OeryruY-P4KuqJeKB6_EXtHUkQ5Cy7u/view?usp=sharing) and a [SiT-XL/2 checkpoint](https://drive.google.com/file/d/1NR_R6wYXS6dwCwYCM8h8EmeLpJjtiwVr/view?usp=sharing) both trained with Dispersive Loss for 80 epochs on ImageNet 256x256.

**Sampling from checkpoint.** To sample from the EMA weights of a 256x256 SiT-XL/2 model checkpoint with ODE sampler, run:
```bash
python sample.py ODE --model SiT-XL/2 --image-size 256 --ckpt /path/to/model.pt
```

### Programmatic Sampling

```python
from dflow.sampling import sample_ode, sample_sde

# ODE sampling
images = sample_ode(
    model, 
    shape=(4, 3, 256, 256), 
    num_steps=100, 
    cfg_scale=1.5
)

# SDE sampling
images = sample_sde(
    model,
    shape=(4, 3, 256, 256),
    num_steps=100,
    cfg_scale=1.5
)
```

**More sampling options.** For more sampling options such as SDE sampling, please refer to [`train_utils.py`](train_utils.py).

## 📊 Evaluation

The [`sample_ddp.py`](sample_ddp.py) script samples a large number of images from a pre-trained model in parallel. This script 
generates a folder of samples as well as a `.npz` file which can be directly used with [ADM's TensorFlow
evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and
other metrics. To sample 50K images from a pre-trained SiT-XL/2 model over `N` GPUs under default ODE sampler settings, run:

```bash
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py ODE --model SiT-XL/2 --num-fid-samples 50000 --ckpt /path/to/model.pt
```

## ⚙️ Configuration

dFlow provides flexible configuration management:

```python
from dflow.configs import get_model_config, get_training_config, get_dispersive_loss_config

# Model configurations
model_config = get_model_config('SiT-XL/2')  # Available: SiT-S/2, SiT-B/2, SiT-L/2, SiT-XL/2, etc.

# Training configurations  
train_config = get_training_config('imagenet_256')

# Dispersive loss configurations
loss_config = get_dispersive_loss_config('default')  # or 'jax_reproduction'
```

## 🔬 Differences from JAX
Our original implementation is in JAX, and this repo contains our re-implementation in PyTorch. 
Therefore, results from running this repo may have minor numerical differences with those reported in our paper. 
In our JAX experiments, we used 16 devices with local batch size 16, whereas in PyTorch experiments we used 8 devices with local batch size 32.
We have adjusted the hyperparameter choices slightly for best performance. 
We report our reproduction results below.

| implementation | config | local bz | B/2 80 ep | XL/2 80 ep (cfg=1.5) |
|-|:-:|:-:|:-:|:-:|
| baseline | - | 16 | 36.49 | 6.02  |
| JAX | $\lambda$=0.5, $\tau$=0.5, depth=num_layers//4 | 16 | 32.35 | 5.09  |
| PyTorch | $\lambda$=0.25, $\tau$=1, depth=num_layers | 32 | 32.64 | 4.74 |

## 📦 Development

### Setup Development Environment

```bash
git clone https://github.com/raywang4/DispLoss.git
cd DispLoss
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black dflow/
flake8 dflow/
```

## 📄 License
This project is under the MIT license. See [LICENSE](LICENSE.txt) for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📖 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{wang2024diffuse,
  title={Diffuse and Disperse: Image Generation with Representation Regularization},
  author={Wang, Runqian and He, Kaiming},
  journal={arXiv preprint arXiv:2506.09027},
  year={2024}
}
```
