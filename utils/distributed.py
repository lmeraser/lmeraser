# Description: This file contains the distributed training utilities.
# Borrowed from [VPT](https://github.com/KMnP/vpt). Thanks to the authors.
# Hacked by this repo's author.

import torch
import torch.distributed as dist
_LOCAL_PROCESS_GROUP = None


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    return 1 if not dist.is_initialized() else dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    return 0 if not dist.is_initialized() else dist.get_rank()


def is_master_process(num_gpus=8):
    """
    Determines if the current process is the master process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() % num_gpus == 0
    else:
        return True
