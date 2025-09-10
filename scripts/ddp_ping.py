# save as ddp_ping.py and run with torchrun --nproc_per_node=4 ddp_ping.py
import os, torch, torch.distributed as dist
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING","1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT","1")
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.barrier()
print(f"Hello from rank={dist.get_rank()} local_rank={local_rank} cuda_count={torch.cuda.device_count()}")
dist.destroy_process_group()
