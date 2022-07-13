# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.optim import SGD, Adam
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
from self_supervised.augmentations import RotationTransform
from self_supervised.augmentations import SimCLRTransform,SimSiamTransform
from self_supervised.criterion import SupConLoss, NTXent, SimSiamLoss


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        self.device = get_device()
        self.rotation_transform = RotationTransform()
        # ssl
        self.init_ssl()

    def init_ssl(self):
        if self.args.ssl_algo == 'supcontrast':
            self.ssl_loss = SupConLoss()
            self.ssl_transform = SimCLRTransform(size=self.args.img_size)
            self.ssl_opt = Adam(self.net.parameters(), lr=3e-4)
        elif self.args.ssl_algo == 'simclr':
            self.ssl_loss = NTXent()
            self.ssl_transform = SimCLRTransform(size=self.args.img_size)
            self.ssl_opt = Adam(self.net.parameters(), lr=3e-4)
        elif self.args.ssl_algo == 'simsiam':
            self.ssl_loss = SimSiamLoss()
            self.ssl_transform = SimSiamTransform(size=self.args.img_size)
            self.ssl_opt = Adam(self.net.parameters(), lr=3e-4, weight_decay=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass

    def rotate(self, inputs: torch.Tensor):
        inputs_rot, labels_rot = [], []
        for input in inputs:
            x, y = self.rotation_transform(input)
            inputs_rot.append(x)
            labels_rot.append(y)
        labels_rot = torch.stack(labels_rot).to(inputs.device)
        inputs_rot = torch.stack(inputs_rot, dim=0).to(inputs.device)
        return inputs_rot, labels_rot

    def ssl_observe(self, inputs: torch.Tensor, labels) -> float:
        self.ssl_opt.zero_grad()

        if self.buffer and not self.buffer.is_empty():
            buf = self.buffer.get_data(self.args.minibatch_size)
            inputs = torch.cat((inputs, buf[0]))
            labels = torch.cat((labels, buf[1]))

        x, y = self.apply_ssl_transform(inputs)

        _, _, zx, px = self.net(x)
        _, _, zy, py = self.net(y)
        if self.args.ssl_algo in ['supcontrast', 'simclr']:
            loss = self.ssl_loss(zx, zy, labels)
        else:
            loss = self.ssl_loss(zx, zy, px, py)
        loss.backward()
        self.ssl_opt.step()

        return loss.item()

    def apply_ssl_transform(self, inputs: torch.Tensor):
        """
        Apply transform for all inputs individually
        """
        X, Y = [], []
        for input in inputs:
            x, y = self.ssl_transform(input)
            X.append(x)
            Y.append(y)
        X = torch.stack(X).to(inputs.device)
        Y = torch.stack(Y, dim=0).to(inputs.device)
        return X, Y
