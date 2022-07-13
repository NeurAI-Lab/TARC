# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from utils.args import *
from models.utils.continual_model import ContinualModel

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning Through'
                                        ' Synaptic Intelligence.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--c', type=float, required=True,
                        help='surrogate loss weight parameter c')
    parser.add_argument('--xi', type=float, required=True,
                        help='xi parameter for EWC online')

    return parser


class SI(ContinualModel):
    NAME = 'si'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(SI, self).__init__(backbone, loss, args, transform)

        self.checkpoint = self.net.get_params().data.clone().to(self.device)
        self.big_omega = None
        self.small_omega = 0

    def penalty(self):
        if self.big_omega is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.big_omega * ((self.net.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def end_task(self, dataset):
        # big omega calculation step
        if self.big_omega is None:
            self.big_omega = torch.zeros_like(self.net.get_params()).to(self.device)

        self.big_omega += self.small_omega / ((self.net.get_params().data - self.checkpoint) ** 2 + self.args.xi)

        # store parameters checkpoint and reset small_omega
        self.checkpoint = self.net.get_params().data.clone().to(self.device)
        self.small_omega = 0

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()

        if self.args.multitask:
            inputs, labels_rot = self.rotate(inputs)
            outputs = self.net(inputs)
            penalty = self.penalty()
            loss_1 = self.loss(outputs[0], labels)
            loss_2 = self.loss(outputs[1], labels_rot)
            loss = self.args.ce_weight * loss_1 + self.args.rot_weight * loss_2 + self.args.c * penalty
            loss.backward()
            nn.utils.clip_grad.clip_grad_value_(self.net.parameters(), 1)
            self.opt.step()
            self.small_omega += self.args.lr * self.net.get_grads().data ** 2
            return loss_1.item(), loss_2.item()
        else:
            outputs = self.net(inputs)
            penalty = self.penalty()
            loss = self.loss(outputs[0], labels) + self.args.c * penalty
            loss.backward()
            nn.utils.clip_grad.clip_grad_value_(self.net.parameters(), 1)
            self.opt.step()
            self.small_omega += self.args.lr * self.net.get_grads().data ** 2
            return loss.item(), torch.tensor(0)
