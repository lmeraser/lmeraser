# Desc: This file is used to setup the logging and environment for the training process.
# Borrowed from [VPT](https://github.com/KMnP/vpt). Thanks to the authors.

import os
import sys
import PIL
from collections import defaultdict
from tabulate import tabulate
from typing import Tuple

import torch


from utils.file_io import PathManager
from utils import logging


def collect_torch_env() -> str:
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # compatible with older versions of pytorch
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


def get_env_module() -> Tuple[str]:
    var_name = "ENV_MODULE"
    return var_name, os.environ.get(var_name, "<not set>")


def collect_env_info() -> str:
    data = [("Python", sys.version.replace("\n", ""))]
    data.extend(
        (
            get_env_module(),
            ("PyTorch", torch.__version__),
            ("PyTorch Debug Build", torch.version.debug),
        )
    )
    has_cuda = torch.cuda.is_available()
    data.append(("CUDA available", has_cuda))
    if has_cuda:
        # if given
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            data.append(("CUDA ID", os.environ["CUDA_VISIBLE_DEVICES"]))
        else:
            data.append(("CUDA ID", torch.cuda.current_device()))
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        data.extend(
            ("GPU " + ",".join(devids), name)
            for name, devids in devices.items()
        )
    data.append(("Pillow", PIL.__version__))

    env_str = tabulate(data) + "\n"
    env_str += collect_torch_env()
    return env_str


def logging_train_setup(args) -> None:
    output_dir = args.output_dir
    if output_dir:
        PathManager.mkdirs(output_dir)

    rank = args.local_rank
    world_size = args.num_gpus

    logger = logging.setup_logging(
        args.num_gpus, output_dir, name="lmeraser", rank=rank
    )

    # Log basic information about environment, cmdline arguments, and config
    
    logger.info(
        f"Rank of current process: {rank}. World size: {world_size}")
    logger.info("Environment info:\n" + collect_env_info())

    logger.info(f"Command line arguments: {str(args)}")
    # cudnn benchmark has large overhead.
    # It shouldn't be used considering the small size of typical val set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = False