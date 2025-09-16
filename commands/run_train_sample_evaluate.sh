export NCCL_P2P_DISABLE=1
export WANDB_KEY="fecb6fa17bef5ff2269bba21500e001f5101fb64"
export ENTITY="mon8c"
export PROJECT="dFlow"

# # Training & Sampling & Evaluating (with noise, cos scheduler)
# current_time=$(date +"%m%d%H%M")
# torchrun --nnodes=1 --nproc_per_node=4 scripts/main.py time=${current_time} noise=true
# python scripts/sampling/sample_hydra.py time=${current_time} noise=true
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=1
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=2
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=4
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=8


# # Training & Sampling & Evaluating (with block_wise_noise, cos scheduler)
# current_time=$(date +"%m%d%H%M")
# torchrun --nnodes=1 --nproc_per_node=4 scripts/main.py time=${current_time} noise=true block_wise_noise=true
# python scripts/sampling/sample_hydra.py time=${current_time} noise=true block_wise_noise=true
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true sampling.num_steps=1
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true sampling.num_steps=2
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true sampling.num_steps=4
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true sampling.num_steps=8


# # Training & Sampling & Evaluating (with block_wise_noisev2, cos scheduler)
# current_time=$(date +"%m%d%H%M")
# torchrun --nnodes=1 --nproc_per_node=4 scripts/main.py time=${current_time} noise=true block_wise_noise=true block_wise_noisev2=true
# python scripts/sampling/sample_hydra.py time=${current_time} noise=true block_wise_noise=true block_wise_noisev2=true
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true block_wise_noisev2=true sampling.num_steps=1
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true block_wise_noisev2=true sampling.num_steps=2
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true block_wise_noisev2=true sampling.num_steps=4
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true block_wise_noisev2=true sampling.num_steps=8

# # Training & Sampling & Evaluating (with noise, square scheduler)
# current_time=$(date +"%m%d%H%M")
# torchrun --nnodes=1 --nproc_per_node=4 scripts/main.py time=${current_time} noise=true model.time_scheduler=square
# python scripts/sampling/sample_hydra.py time=${current_time} noise=true model.time_scheduler=square
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=1 model.time_scheduler=square
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=2 model.time_scheduler=square
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=4 model.time_scheduler=square
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=8 model.time_scheduler=square


# # Training & Sampling & Evaluating (with block_wise_noise, square scheduler)
# current_time=$(date +"%m%d%H%M")
# torchrun --nnodes=1 --nproc_per_node=4 scripts/main.py time=${current_time} noise=true block_wise_noise=true model.time_scheduler=square
# python scripts/sampling/sample_hydra.py time=${current_time} noise=true block_wise_noise=true model.time_scheduler=square
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true sampling.num_steps=1 model.time_scheduler=square
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true sampling.num_steps=2 model.time_scheduler=square
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true sampling.num_steps=4 model.time_scheduler=square
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true sampling.num_steps=8 model.time_scheduler=square


# # Training & Sampling & Evaluating (with block_wise_noisev2, square scheduler)
# current_time=$(date +"%m%d%H%M")
# torchrun --nnodes=1 --nproc_per_node=4 scripts/main.py time=${current_time} noise=true block_wise_noise=true block_wise_noisev2=true model.time_scheduler=square
# python scripts/sampling/sample_hydra.py time=${current_time} noise=true block_wise_noise=true block_wise_noisev2=true model.time_scheduler=square
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true block_wise_noisev2=true sampling.num_steps=1 model.time_scheduler=square
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true block_wise_noisev2=true sampling.num_steps=2 model.time_scheduler=square
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true block_wise_noisev2=true sampling.num_steps=4 model.time_scheduler=square
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true block_wise_noisev2=true sampling.num_steps=8 model.time_scheduler=square

# # Training & Sampling & Evaluating (with block_wise_noise, cos scheduler, avg_vf=2)
# current_time=$(date +"%m%d%H%M")
# torchrun --nnodes=1 --nproc_per_node=4 scripts/main.py time=${current_time} noise=true block_wise_noise=true model.avg_vf=2 dataset.loader.batch_size=512
# python scripts/sampling/sample_hydra.py time=${current_time} noise=true block_wise_noise=true
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true sampling.num_steps=1 model.avg_vf=2
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true sampling.num_steps=2 model.avg_vf=2
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true sampling.num_steps=4 model.avg_vf=2
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true sampling.num_steps=8 model.avg_vf=2

# # Training & Sampling & Evaluating (with block_wise_noise, cos scheduler, avg_vf=2, variance loss=0.1)
# current_time=$(date +"%m%d%H%M")
# torchrun --nnodes=1 --nproc_per_node=4 scripts/main.py time=${current_time} noise=true block_wise_noise=true model.avg_vf=2 dataset.loader.batch_size=512 training.additional_loss=true training.var_loss=0.1
# python scripts/sampling/sample_hydra.py time=${current_time} noise=true block_wise_noise=true
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true sampling.num_steps=1 model.avg_vf=2
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true sampling.num_steps=2 model.avg_vf=2
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true sampling.num_steps=4 model.avg_vf=2
# torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true block_wise_noise=true sampling.num_steps=8 model.avg_vf=2

# Training & Sampling & Evaluating (from 600 epochs PT model, with noise, cos scheduler, lr:1e-4)
current_time=$(date +"%m%d%H%M")
torchrun --nnodes=1 --nproc_per_node=4 scripts/main.py time=${current_time} noise=true training.checkpoint=/storage/sunil/exp/dFlow/ckpts/final.pt training.max_epochs=30 optimizer.lr=0.0001
python scripts/sampling/sample_hydra.py time=${current_time} noise=true
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=1
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=2
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=4
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=8

# Training & Sampling & Evaluating (from 600 epochs PT model, with noise, cos scheduler, lr:1e-5)
current_time=$(date +"%m%d%H%M")
torchrun --nnodes=1 --nproc_per_node=4 scripts/main.py time=${current_time} noise=true training.checkpoint=/storage/sunil/exp/dFlow/ckpts/final.pt training.max_epochs=30 optimizer.lr=0.00001
python scripts/sampling/sample_hydra.py time=${current_time} noise=true
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=1
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=2
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=4
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=8

# Training & Sampling & Evaluating (from 600 epochs PT model, with noise, cos scheduler, lr:1e-6)
current_time=$(date +"%m%d%H%M")
torchrun --nnodes=1 --nproc_per_node=4 scripts/main.py time=${current_time} noise=true training.checkpoint=/storage/sunil/exp/dFlow/ckpts/final.pt training.max_epochs=30 optimizer.lr=0.000001
python scripts/sampling/sample_hydra.py time=${current_time} noise=true
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=1
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=2
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=4
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=8

# Training & Sampling & Evaluating (from 600 epochs PT model, with noise, square scheduler, lr:1e-4)
current_time=$(date +"%m%d%H%M")
torchrun --nnodes=1 --nproc_per_node=4 scripts/main.py time=${current_time} noise=true training.checkpoint=/storage/sunil/exp/dFlow/ckpts/final.pt training.max_epochs=30 optimizer.lr=0.0001 model.time_scheduler=square
python scripts/sampling/sample_hydra.py time=${current_time} noise=true model.time_scheduler=square
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=1 model.time_scheduler=square
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=2 model.time_scheduler=square
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=4 model.time_scheduler=square
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=8 model.time_scheduler=square

# Training & Sampling & Evaluating (from 600 epochs PT model, with noise, square scheduler, lr:1e-5)
current_time=$(date +"%m%d%H%M")
torchrun --nnodes=1 --nproc_per_node=4 scripts/main.py time=${current_time} noise=true training.checkpoint=/storage/sunil/exp/dFlow/ckpts/final.pt training.max_epochs=30 optimizer.lr=0.00001 model.time_scheduler=square
python scripts/sampling/sample_hydra.py time=${current_time} noise=true model.time_scheduler=square
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=1 model.time_scheduler=square
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=2 model.time_scheduler=square
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=4 model.time_scheduler=square
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=8 model.time_scheduler=square

# Training & Sampling & Evaluating (from 600 epochs PT model, with noise, square scheduler, lr:1e-6)
current_time=$(date +"%m%d%H%M")
torchrun --nnodes=1 --nproc_per_node=4 scripts/main.py time=${current_time} noise=true training.checkpoint=/storage/sunil/exp/dFlow/ckpts/final.pt training.max_epochs=30 optimizer.lr=0.000001 model.time_scheduler=square
python scripts/sampling/sample_hydra.py time=${current_time} noise=true model.time_scheduler=square
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=1 model.time_scheduler=square
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=2 model.time_scheduler=square
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=4 model.time_scheduler=square
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time} noise=true sampling.num_steps=8 model.time_scheduler=square