# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from datasets import get_gcl_dataset
from models import get_model
from utils.status import progress_bar
from utils.tb_logger import *
from utils.status import create_fake_stash
from models.utils.continual_model import ContinualModel
from argparse import Namespace


def evaluate(model: ContinualModel, dataset) -> float:
    """
    Evaluates the final accuracy of the model.
    :param model: the model to be evaluated
    :param dataset: the GCL dataset at hand
    :return: a float value that indicates the accuracy
    """
    model.net.eval()
    correct, total = 0, 0
    while not dataset.test_over:
        inputs, labels = dataset.get_test_data()
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs[0].data, 1)
        correct += torch.sum(predicted == labels).item()
        total += labels.shape[0]

    acc = correct / total * 100
    return acc


def train(args: Namespace):
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """

    dataset = get_gcl_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())
    model.net.to(model.device)

    model_stash = create_fake_stash(model, args)

    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        # from utils.loggers import CsvLogger
        # csv_logger = CsvLogger(tb_logger.get_log_dir())

    model.net.train()
    epoch, i = 0, 0
    ssl_loss = 0
    while not dataset.train_over:
        inputs, labels, not_aug_inputs = dataset.get_train_data()
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        not_aug_inputs = not_aug_inputs.to(model.device)
        if args.train_ssl  and dataset.completed_rounds < 3:
            ssl_loss += model.ssl_observe(not_aug_inputs, labels)
        else:
            loss, loss_rot = model.observe(inputs, labels, not_aug_inputs)
            if args.tensorboard:
                tb_logger.log_loss_gcl(loss, i)
            progress_bar(i, dataset.LENGTH // args.batch_size, epoch, 'C', loss)

        i += 1


    if model.NAME == 'joint_gcl':
      model.end_task(dataset)

    acc = evaluate(model, dataset)
    print('Accuracy:', acc)

    try:
        f = open(os.path.join(tb_logger.get_log_dir(), 'ece.txt'), "w")
        f.write('Accuracy score: {}'.format(acc))
        f.close()
    except:
        raise Exception('Unable to write accuracy to a file!')
