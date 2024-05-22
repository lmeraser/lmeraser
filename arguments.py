# Description: This file contains the arguments for training and testing.
# Moidified from [DAM-VP](https://github.com/shikiw/DAM-VP)

import argparse

MODEL_LIST = [
    "vit-b-22k", 
    "vit-b-1k", 
    "swin-b-22k", 
    "swin-b-1k", 
]

ERASE_DATASETS = [
    "cifar10", 
    "cifar100", 
    "svhn", 
    "gtsrb",
]

DATASET_DIVERSITIES = {
    "cifar10": 70.2, 
    "cifar100": 70.9, 
    "svhn": 61.8, 
    "gtsrb": 67.5,
}


class Arguments:
    def __init__(self, stage='eraser', distributed=False):
        self._parser = argparse.ArgumentParser(description='Diversity-Aware Meta Visual Prompting.')
        self.add_args()

    def add_args(self):
        ### log related
        self.add_argument('--output_dir', type=str, default='')

        ### data related
        self.add_argument('--batch_size', type=int, default=128, help='Batch size in training')
        self.add_argument('--base_dir', type=str, default='../')
        self.add_argument('--crop_size', default=224, type=int, help='Input size of images [default: 224].')
        self.add_argument('--diversities', type=dict, default=DATASET_DIVERSITIES, help='Diversity values of datasets.')

        ### prompt related
        self.add_argument('--pretrained_model', type=str, default='vit-b-22k', choices=MODEL_LIST)
        self.add_argument('--prompt_method', type=str, default='padding', choices=['padding', 'fixed_patch', 'random_patch'])
        self.add_argument('--prompt_size', type=int, default=30, help='Padding size for visual prompts.')
        self.add_argument('--one_prompt', action='store_true', default=False, help='Without diversity-aware strategy [default: False].')

        ### model related
        self.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
        self.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT', help='Attention dropout rate (default: 0.)')
        self.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')

        ### others
        self.add_argument('--seed', type=int, default=2023, metavar='S', help='Random seed (default: 2023)')
        self.add_argument('--gpu_ids', type=int, default=0, help='Ids of GPUs to use.')
        self.add_argument('--num_gpus', type=int, default=1, help='Num of GPUs to use.')
        self.add_argument('--num_workers', type=int, default=4, help='Worker nums of data loading.')
        self.add_argument('--pin_memory', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
        self.parser().set_defaults(pin_memory=True)
        self.add_argument('--distributed', action='store_true', default=False, help='Whether to use the distributed mode [default: False].')
        self.add_argument('--training_usage_percentage', type=float, default=1.0, help='Percentage of training data for usage.')

        self.add_argument('--distance_threshold', type=float, default=10, help='Threshold for distance.')
        self.add_argument('--test_dataset', type=str, default='oxford-flowers', choices=ERASE_DATASETS, help='The dataset selected for evaluation')
        self.add_argument('--erasing_method', type=str, default='prompt_wo_head', choices=['lmeraser', 'random_part_tuning'])
        self.add_argument('--checkpoint_dir', type=str, default='')

        ### tuning related
        self.add_argument('--epochs', type=int, default=50, help='Nums of training epochs.')
        self.add_argument('--optimizer', type=str, default='Adam', help='Optimizer for training [default: Adam]')
        self.add_argument('--lr', type=float, default=1e+4, help='Learning rate')
        self.add_argument('--weight_decay', type=float, default=0, help='Weight decay rate')
        self.add_argument('--eval_only', action='store_true', default=False, help='Evaluate only [default: False].')


    def parser(self):
        return self._parser

    def add_argument(self, *args, **kwargs):
        self.parser().add_argument(*args, **kwargs)
