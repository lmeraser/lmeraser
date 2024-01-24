# Description: Data loader for CIFAR10, CIFAR100, SVHN, GTSRB
# Modified from [DAM-VP](https://github.com/shikiw/DAM-VP). Thanks to the authors.

import os
import sys
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))

from utils import logging
from data_utils.datasets import CIFAR10, CIFAR100, SVHN, GTSRB

logger = logging.get_logger("lmeraser")
_DATASET_CATALOG = {
    "cifar10": CIFAR10, 
    "cifar100": CIFAR100, 
    "svhn": SVHN, 
    "gtsrb": GTSRB
}

_DATA_DIR_CATALOG = {
    "cifar10": "torchvision_dataset/", 
    "cifar100": "torchvision_dataset/", 
    "svhn": "torchvision_dataset/", 
    "gtsrb": "torchvision_dataset/"
}

_NUM_CLASSES_CATALOG = {
    "cifar10": 10, 
    "cifar100": 100, 
    "svhn": 10, 
    "gtsrb": 43
}


def get_dataset_classes(dataset):
    """Given a dataset, return the name list of dataset classes."""
    if hasattr(dataset, "classes"):
        return dataset.classes
    elif hasattr(dataset, "_class_ids"):
        return dataset._class_ids
    elif hasattr(dataset, "labels"):
        return dataset.labels
    else:
        raise NotImplementedError


def _construct_loader(args, dataset, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    dataset_name = dataset


    assert (
        dataset_name in _DATASET_CATALOG.keys()
    ), f"Dataset '{dataset_name}' not supported"
    args.data_dir = os.path.join(args.base_dir, _DATA_DIR_CATALOG[dataset_name])
    dataset = _DATASET_CATALOG[dataset_name](args, split, download=True, sub_percentage=args.training_usage_percentage)

    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if args.distributed else None

    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=drop_last,
    )
    if args.pretrained_model.startswith("clip-"):# and args.erasing_method in ["vp", "ours"]:
        return loader, get_dataset_classes(dataset)
    return loader


def construct_train_loader(args, dataset=None):
    """Train loader wrapper."""
    drop_last = bool(args.distributed)
    return _construct_loader(
        args=args,
        split="train",
        batch_size=int(args.batch_size / args.num_gpus),
        shuffle=True,
        drop_last=drop_last,
        dataset=dataset or args.dataset
    )


def construct_val_loader(args, dataset=None, batch_size=None):
    """Validation loader wrapper."""
    bs = int(args.batch_size / args.num_gpus) if batch_size is None else batch_size
    return _construct_loader(
        args=args,
        split="val",
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        dataset=dataset or args.dataset
    )


def construct_test_loader(args, dataset=None):
    """Test loader wrapper."""
    return _construct_loader(
        args=args,
        split="test",
        batch_size=int(args.batch_size / args.num_gpus),
        shuffle=False,
        drop_last=False,
        dataset=dataset or args.dataset
    )


def shuffle(loader, cur_epoch):
    """"Shuffles the data."""
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), f"Sampler type '{type(loader.sampler)}' not supported"
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)


def _dataset_class_num(dataset_name):
    """Query to obtain class nums of datasets."""
    return _NUM_CLASSES_CATALOG[dataset_name]
