###
 # File: eraser/eraser.py
 # Created Date: Dec 16th 2023
 # Author: Anonymous
 # -----
 # Last Modified: Saturday, 20th January 2024 1:02:18 am
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

from arguments import Arguments
import utils.logging as logging
from utils.lr import cosine_lr
from data_utils import loader as data_loader
from models import prompters
import os
import sys
import numpy as np
import os.path as osp
from copy import deepcopy

import torch
import torch.nn as nn
import sklearn.cluster as cluster


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))


logger = logging.get_logger("lmeraser")


def hash_tensor(tensor, num_classes, seed=2024):
    """
    Hashes a tensor and returns the hash value modulo the specified number of classes.

    Args:
        tensor: The tensor to be hashed.
        num_classes: The number of classes to modulo the hash value by.
        seed: The seed value for generating the same hash result for the same tensor.

    Returns:
        int: The hash value modulo the number of classes.

    Examples:
        >>> tensor = torch.tensor([[1, 2], [3, 4]])
        >>> hash_tensor(tensor, 10, seed=2024)
        5
    """

    import hashlib

    # ensure the same hash result for the same tensor
    torch.manual_seed(seed)

    # flatten the tensor
    flattened_tensor = tensor.flatten()

    # convert the tensor to bytes
    tensor_bytes = flattened_tensor.to('cpu').numpy().tobytes()

    # hash the bytes
    hash_object = hashlib.sha256(tensor_bytes)
    hash_digest = hash_object.hexdigest()

    return int(hash_digest, 16) % num_classes

class Eraser(object):
    """
    A class representing the Eraser.

    Args:
        args: An instance of Arguments class.
        model: An instance of nn.Module class.

    Raises:
        None

    Examples:
        >>> args = Arguments()
        >>> model = nn.Module()
        >>> eraser = Eraser(args, model)
    """

    def __init__(self, args: Arguments, model: nn.Module):
        """
        Initializes an instance of the Eraser class.

        Args:
            args: The Arguments instance containing the arguments.
            model: The nn.Module instance representing the model.

        Raises:
            None
        """

        super(Eraser, self).__init__()
        self.args = args
        self.model = model.eval()

        self.logger = logging.get_logger("lmeraser")

        self.lr = args.lr
        self.weight_decay = args.weight_decay

        self.device_type = args.device_type # "cuda" or "cpu" or "mps"
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else None

        # self.logger.info(f"device_type: {self.args.device_type}")
        # self.logger.info(f"self.rank: {self.rank}")

        self.devicename = torch.device(
            f"{self.device_type}:{self.rank}" if self.rank is not None else self.device_type
        )
        self.criterion = torch.nn.CrossEntropyLoss().to(self.devicename)

    def loss(self, logits, target):
        return self.criterion(logits, target)

    def load_prompter(self, prompter_path=None):
        """Load the trained visual prompter from files
        """
        prompter = prompters.__dict__[self.args.prompt_method](
            self.args).to(self.devicename) # prompt_method: padding, fixed_patch, random_patch
        if prompter_path is not None:
            checkpoint = torch.load(
                prompter_path, map_location=self.devicename)
            prompter.load_state_dict(checkpoint['state_dict'])
            logger.info(f"Loading meta-trained visual prompts from {prompter_path}")
        return prompter

    def get_prompted_image(self, image, prototype_gather=None, prompter=None, prompter_gather=None):
        """
        Obtains the prompted batch images based on the given inputs.

        Args:
            self: The instance of the class.
            image: The input image tensor.
            prototype_gather: The tensor containing prototype gather.
            prompter: The prompter function.
            prompter_gather: The list of prompter functions.

        Returns:
            Tuple: A tuple containing the indices tensor and the prompted image tensor.

        Raises:
            AssertionError: If the prompter or prompter_gather is None when one_prompt is False.

        Borrowed from [DAM-VP](https://github.com/shikiw/DAM-VP) and hacked by this repo's author.
    
        Thanks to the authors.
        """

        if self.args.one_prompt:
            assert prompter is not None
            prompted_image = prompter(image)
            indices = None
        else:
            assert prototype_gather is not None
            assert prompter_gather is not None
            prompted_image = []
            with torch.no_grad():
                rep_batch = self.model.forward_features(image)  # [N, emd_dim]
                rep_batch_sum = (rep_batch**2).sum(dim=-1,
                                                   keepdims=True)  # [N, 1]
                prototype_gather_sum = (
                    prototype_gather**2).sum(dim=-1, keepdims=True).T  # [1, M]
                distance_matrix = torch.sqrt(
                    rep_batch_sum + prototype_gather_sum - 2 * torch.mm(rep_batch, prototype_gather.T))  # [N, M]
                indices = torch.argmin(distance_matrix, dim=-1)  # [B]

            prompted_image = [
                prompter_gather[indices[idx]](image[idx].unsqueeze(0))
                for idx in range(rep_batch.size(0))
            ]

            prompted_image = torch.cat(prompted_image, dim=0) # [B, 3, 224, 224]

        return indices, prompted_image

    def coarse_clustering(self, data_loader):
        """
        Performs coarse clustering on the data loader.

        Args:
            self: The instance of the class.
            data_loader: The data loader containing the training data.

        Returns:
            None

        Raises:
            None

        Borrowed from [DAM-VP](https://github.com/shikiw/DAM-VP). Thanks to the authors.
        """


        # if saved exists, load it
        if osp.exists(f"{self.args.output_dir}/prototype_gather_{self.devicename}.pth") and osp.exists(f"{self.args.output_dir}/num_coarse_classes_{self.devicename}.pth"):
            logger.info(
                f"Loading prototype gather from {self.args.output_dir}/prototype_gather_{self.devicename}.pth")
            save_dict = torch.load(
                f"{self.args.output_dir}/prototype_gather_{self.devicename}.pth")
            self.prototype_gather = save_dict["prototype_gather"]
            self.num_coarse_classes = save_dict["num_coarse_classes"]
            return

        train_loader, _, _ = data_loader
        threshold_dict = {
            "vit-b-22k": self.args.distance_threshold,
            "swin-b-22k": 20,
        }
        hc = cluster.AgglomerativeClustering(
            n_clusters=None,
            linkage='average',
            distance_threshold=threshold_dict[self.args.pretrained_model]
        )
        rep_gather = None  # Initialize rep_gather
        with torch.no_grad():
            for sample in train_loader:
                image = sample["image"].to(self.devicename)
                rep = self.model.forward_features(image)
                rep_gather = rep if rep_gather is None else torch.cat([rep_gather, rep], dim=0)
                if rep_gather.size(0) > 1000:
                    rep_gather = rep_gather[:1000]
                    break

        y_pred = hc.fit(rep_gather.detach().cpu().numpy()).labels_
        y_pred = torch.from_numpy(y_pred).to(self.devicename)
        coarse_class_idx = torch.unique(y_pred)
        self.num_coarse_classes = len(coarse_class_idx)
        logger.info(
            f"Nums of coarsely divided categories for test dataset {self.args.test_dataset}: {len(coarse_class_idx)}"
        )

        prototype_gather = []
        for i in range(len(coarse_class_idx)):
            pos = torch.where(y_pred == i)[0]
            prototype = rep_gather[pos].mean(0).unsqueeze(0)
            prototype_gather.append(prototype)
        self.prototype_gather = torch.cat(prototype_gather)
        logger.info(
            f"Nums of prototypes of coarse clusters for test dataset {self.args.test_dataset}: {self.prototype_gather.size(0)}"
        )
        # save_dict = {
        #     "prototype_gather": self.prototype_gather,
        #     "num_coarse_classes": self.num_coarse_classes
        # }
        # torch.save(save_dict, f"{self.args.output_dir}/prototype_gather_{self.devicename}.pth")

    def LMErasing_tuning(self, test_data, prompter_path):
        """
        Performs LMErasing tuning on the given test data using the specified prompter.

        Args:
            self: The instance of the class.
            test_data: The test data containing train, validation, and test loaders.
            prompter_path: The path to the prompter.

        Returns:
            int: The result of the LMErasing tuning.

        Raises:
            None
        """

        logger.info(f"Start lmerasing_with_mul_head on {self.devicename}")
        train_loader, val_loader, test_loader = test_data
        prompter = self.load_prompter(prompter_path)
        num_classes = data_loader._dataset_class_num(self.args.test_dataset) # class from label
        self.model.reset_classifier(num_classes) # define new head

        if not self.args.one_prompt:
            self.coarse_clustering(test_data) # cluster from diversity aware
            # updated self.num_coarse_classes
            # updated self.prototype_gather

        self.set_head_list()
        if self.args.one_prompt:
            # prompter = deepcopy(prompter)
            optimizer = torch.optim.SGD([
                {'params': prompter.parameters(), 'lr': self.lr, 'momentum': 0.9,
                 'weight_decay': self.weight_decay},
                {'params': self.model.get_classifier().parameters(), 'lr': 0.1,
                 'momentum': 0.9, 'weight_decay': 0}
            ])
        else:
            # head_params = [
            #     param for head in self.head_list for param in head.parameters()]

            prompter_gather, prompter_params_gather = [], []
            for i in range(self.num_coarse_classes):
                prompter_gather.append(deepcopy(prompter))
                prompter_params_gather.extend(
                    (
                        {
                            'params': prompter_gather[i].parameters(),
                            'lr': self.lr,
                            'momentum': 0.9,
                            'weight_decay': self.weight_decay,
                        },
                        {
                            'params': self.head_list[i].parameters(),
                            'lr': 0.1,
                            'momentum': 0.9,
                            'weight_decay': 0,
                        },
                    )
                )

            optimizer = torch.optim.SGD(prompter_params_gather)

        scheduler = cosine_lr(
            optimizer,
            self.lr,
            len(train_loader) * self.args.epochs // 5,
            len(train_loader) * self.args.epochs
        )

        BEST_ACC_VAL = -np.inf
        best_prompter_gather = deepcopy(prompter_gather)

        for epoch in range(self.args.epochs):
            # train
            for i, sample in enumerate(train_loader):
                # adjust learning rate
                loss, acc_train = self.training(train_loader, prompter, optimizer, prompter_gather, i, scheduler, epoch, sample)
                logger.info(
                    f"[Prompt Finetuning] Epoch: [{epoch}/{self.args.epochs}], Step: [{i}/{len(train_loader)}], Training loss: {loss.item()}, Training acc: {acc_train}"
                )

            # validate
            with torch.no_grad():
                acc_val, loss_val = self.evaluation(val_loader, prompter, prompter_gather)
                logger.info(
                    f"[Prompt Validating] Epoch: {epoch}, Val acc: {acc_val}, Val loss: {loss_val}")
                if acc_val > BEST_ACC_VAL:
                    BEST_ACC_VAL = acc_val
                    best_prompter_gather = deepcopy(prompter_gather)

            if epoch > 0 and (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    acc_test, loss_test = self.evaluation(test_loader, prompter, best_prompter_gather)
                    logger.info(
                        f"[Prompt Testing] Epoch: {epoch}, Test acc: {acc_test}, Test loss: {loss_test}")
        return 0

    def training(self, train_loader, prompter, optimizer, prompter_gather, i, scheduler, epoch, sample):
        """
        Performs training on the given train loader using the specified prompter.

        Args:
            self: The instance of the class.
            train_loader: The train loader containing the training data.
            prompter: The prompter function.
            optimizer: The optimizer for training.
            prompter_gather: The list of prompter functions.
            i: The current step index.
            scheduler: The scheduler for adjusting the learning rate.
            epoch: The current epoch.
            sample: The current sample from the train loader.

        Returns:
            Tuple: A tuple containing the loss and the accuracy of the training.

        Raises:
            None
        """
        global_step = len(train_loader) * epoch + i
        scheduler(global_step)
        image = sample["image"].to(self.devicename)
        label = sample["label"].to(self.devicename)
                # label[:int(self.args.batch_size/10)] = int(torch.randint(0, num_classes, (1,)).item())
        logits, loss = self.infer(
                    prompter, prompter_gather, image, label)

        self.train_step(optimizer, loss)
        act_batch_size = image.size(0)
        pred = torch.argmax(logits, dim=-1)
        correct = (pred == label).sum().item()
        acc_train = float(correct / act_batch_size)
        return loss,acc_train

    def evaluation(self, test_loader, prompter, best_prompter_gather):
        """
        Performs evaluation on the given test loader using the specified prompter and best_prompter_gather.

        Args:
            self: The instance of the class.
            test_loader: The test loader containing the test data.
            prompter: The prompter function.
            best_prompter_gather: The best prompter gather.

        Returns:
            [accuarcy, loss]: A list containing the accuracy and the loss of the evaluation.

        Raises:
            None
        """
        num_total, correct, loss_total = 0, 0, 0
        for sample in test_loader:
            image = sample["image"].to(self.devicename)
            label = sample["label"].to(self.devicename)
            logits, loss = self.infer(
                            prompter, best_prompter_gather, image, label)
            loss_total += loss.item()

            pred = torch.argmax(logits, dim=-1)
            correct += (pred == label).sum().item()
            num_total += image.size(0)
        acc_test = float(correct / num_total)
        loss_test = float(loss_total / len(test_loader))
        return acc_test,loss_test

    def random_part_tuning(self, test_data, prompter_path):
        """
        Performs random partition tuning on the given test data using the specified prompter.
        
        Args:
            self: The instance of the class.
            test_data: The test data containing train, validation, and test loaders.
            prompter_path: The path to the prompter.
            
        Returns:
            int: The result of the random partition tuning.
            
        Raises:
            None
        """
        logger.info(f"Start random partition on {self.devicename}")
        train_loader, val_loader, test_loader = test_data
        prompter = self.load_prompter(prompter_path)
        num_classes = data_loader._dataset_class_num(self.args.test_dataset)
        self.model.reset_classifier(num_classes)

        self.coarse_clustering(test_data)
        self.set_head_list()
        prompter_gather, prompter_params_gather = [], []
        for i in range(self.num_coarse_classes):
            prompter_gather.append(deepcopy(prompter))
            prompter_params_gather.extend(
                (
                    {
                        'params': prompter_gather[i].parameters(),
                        'lr': self.lr,
                        'momentum': 0.9,
                        'weight_decay': self.weight_decay,
                    },
                    {
                        'params': self.head_list[i].parameters(),
                        'lr': 0.1,
                        'momentum': 0.9,
                        'weight_decay': 0,
                    },
                )
            )
        optimizer = torch.optim.SGD(prompter_params_gather)

        scheduler = cosine_lr(
            optimizer,
            self.lr,
            len(train_loader) * self.args.epochs // 5,
            len(train_loader) * self.args.epochs
        )

        BEST_ACC_VAL = -np.inf

        for epoch in range(self.args.epochs):
            # train
            for i, sample in enumerate(train_loader):
                # adjust learning rate
                self.train_rand(train_loader, prompter, prompter_gather, i, optimizer, scheduler, epoch, sample)

            # validate
            with torch.no_grad():
                acc_val, loss_val = self.evaluation_rand(val_loader, prompter, prompter_gather)
                logger.info(
                    f"[Prompt Validating] Epoch: {epoch}, Val acc: {acc_val}, Val loss: {loss_val}")
                if acc_val > BEST_ACC_VAL:
                    BEST_ACC_VAL = acc_val
                    best_prompter_gather = deepcopy(prompter_gather)

            # test
            # torch.save(best_prompter_gather,
            #            f"{self.args.output_dir}/best_prompter_gather_{self.devicename}_epoch_{epoch}.pth")
            # logger.info(
            #     f"Saving best prompter gather to {self.args.output_dir}/best_prompter_gather_{self.devicename}_epoch_{epoch}.pth")
            # save model head
            # self.head = nn.Linear(self.num_features, num_classes)
            # torch.save(self.head_list,
            #            f"{self.args.output_dir}/head_list_{self.devicename}_epoch_{epoch}.pth")
            # logger.info(
            #     f"Saving model head to {self.args.output_dir}/head_list_{self.devicename}_epoch_{epoch}.pth")

            if epoch > 0 and (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    acc_test, loss_test = self.evaluation_rand(test_loader, prompter, best_prompter_gather)
                    logger.info(
                        f"[Prompt Testing] Epoch: {epoch}, Test acc: {acc_test}, Test loss: {loss_test}")
        return 0

    def set_head_list(self):
        self.head_list = self.model.get_multi_classifier(self.num_coarse_classes)
        self.head_list = [head.to(self.devicename) for head in self.head_list]
        for head in self.head_list:
            head.train()

    def train_rand(self, train_loader, prompter, prompter_gather, step, optimizer, scheduler, epoch, batch):
        global_step = len(train_loader) * epoch + step
        scheduler(global_step)
        image = batch["image"].to(self.devicename)
        label = batch["label"].to(self.devicename)
        logits, loss = self.infer_rand(
                    prompter, prompter_gather, image, label)

        self.train_step(optimizer, loss)
        act_batch_size = image.size(0)
        pred = torch.argmax(logits, dim=-1)
        correct = (pred == label).sum().item()
        acc_train = float(correct / act_batch_size)
        logger.info(
                    f"[Prompt Finetuning] Epoch: [{epoch}/{self.args.epochs}], Step: [{step}/{len(train_loader)}], Training loss: {loss.item()}, Training acc: {acc_train}"
                )

    def train_step(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def evaluation_rand(self, test_loader, prompter, prompter_gather):
        """
        Performs evaluation on the given test loader, random partitioning version.

        Args:
            self: The instance of the class.
            test_loader: The test loader containing the test data.
            prompter: The prompter function.
            prompter_gather: The prompter gather.

        Returns:
            [accuarcy, loss]: A list containing the accuracy and the loss of the evaluation.
        
        """
        num_total, correct, loss_total = 0, 0, 0
        for sample in test_loader:
            image = sample["image"].to(self.devicename)
            label = sample["label"].to(self.devicename)
            logits, loss = self.infer_rand(
                            prompter, prompter_gather, image, label)
            loss_total += loss.item()

            pred = torch.argmax(logits, dim=-1)
            correct += (pred == label).sum().item()
            num_total += image.size(0)
        acc_test = float(correct / num_total)
        loss_test = float(loss_total / len(test_loader))
        return acc_test,loss_test

    def infer(self, prompter, prompter_gather, image, label):
        """
        Infers the logits and loss on the given image and label using the specified prompter and prompter_gather.

        Args:
            self: The instance of the class.
            prompter: The prompter function.
            prompter_gather: The prompter gather.
            image: The image tensor.
            label: The label tensor.

        Returns:
            Tuple: A tuple containing the logits and the loss.

        Raises:
            None
        
        """
        indices, prompted_image = self.get_prompted_image(image, prompter=prompter) \
            if self.args.one_prompt else self.get_prompted_image(image, self.prototype_gather, prompter_gather=prompter_gather)
        reps = self.model.forward_features(prompted_image)
        # logits = self.head_list[indices](reps)

        logits = torch.stack([self.head_list[indices[idx]](reps[idx])
                              for idx in range(len(indices))], dim=0)

        loss = self.loss(logits, label)
        return logits, loss

    def infer_rand(self, prompter, prompter_gather, image, label):
        """
        Infers the logits and loss on the given image and label using the specified prompter and prompter_gather.
        Random partitioning version.

        Args:
            self: The instance of the class.
            prompter: The prompter function.
            prompter_gather: The prompter gather.
            image: The image tensor.
            label: The label tensor.

        Returns:
            Tuple: A tuple containing the logits and the loss.
        
        """
        indices, prompted_image = self.get_prompted_image_rand(
            image, prompter_gather=prompter_gather, seed=2024)

        reps = self.model.forward_features(prompted_image)
        # logits = self.head_list[indices](reps)

        logits = torch.stack([self.head_list[indices[idx]](reps[idx])
                              for idx in range(len(indices))], dim=0)

        loss = self.loss(logits, label)
        return logits, loss

    def get_prompted_image_rand(self, image, prompter_gather, seed=2024):
        """Obtain the prompted batch images.
        """

        assert self.num_coarse_classes == len(prompter_gather)

        hash_list = [
            hash_tensor(image[idx], len(prompter_gather), seed=seed)
            for idx in range(image.size(0))
        ]

        indices = torch.tensor(hash_list).to(self.devicename)

        prompted_image = [
            prompter_gather[indices[idx]](image[idx].unsqueeze(0))
            for idx in range(image.size(0))
        ]

        return indices, torch.cat(prompted_image, dim=0)
