export NCCL_P2P_DISABLE=1
export WANDB_KEY="fecb6fa17bef5ff2269bba21500e001f5101fb64"
export ENTITY="mon8c"
export PROJECT="dFlow"

current_time=$(date +"%m%d%H%M")

# Training & Sampling & Evaluating (with noise)
torchrun --nnodes=1 --nproc_per_node=4 scripts/main.py time=${current_time} noise=true
python scripts/sampling/sample_hydra.py time=${current_time} noise=true
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true

# # Training & Sampling & Evaluating (without noise)
# torchrun --nnodes=1 --nproc_per_node=4 scripts/main.py time=${current_time}
# python scripts/sampling/sample_hydra.py time=${current_time}
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time}