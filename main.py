# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import sys
conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/models')

from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import ContinualDataset
from utils.continual_training import train as ctrain
from datasets import get_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def main():
    lecun_fix()
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    parser.add_argument('--train_ssl', action='store_true',
                        help='Use SSL for task agnostic learning.')
    parser.add_argument('--ssl_algo', type=str, default='supcontrast',
                        help='SSL training algorithm')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone after SSL training and train CL only using single linear layer')
    parser.add_argument('--multitask', action='store_true',
                        help='Use rotation as multitask')
    parser.add_argument('--empty_buffer', action='store_true',
                        help='Do not use any buffer for experience replay')
    parser.add_argument('--ce_weight', type=float, default=1,
                        help='multitask weight for cross entropy')
    parser.add_argument('--rot_weight', type=float, default=1,
                        help='multitask weight for rotation')
    parser.add_argument('--img_size', type=int, required=True,
                        help='Input image size')
    parser.add_argument('--ssl_train_percentage', type=float, default=0.6,
                        help='Percentage of training time for SSL. Rest is for multitasking')
    parser.add_argument('--eval_c', action='store_true',
                        help='Use trained model for evaluation on natural corruption datasets')
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=DATASET_NAMES,
                            help='Which dataset to perform experiments on.')
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'
        if hasattr(args, 'buffer_size'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        for key, value in best.items():
            setattr(args, key, value)
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        parser.add_argument('--train_ssl', action='store_true',
                            help='Use SSL for task agnostic learning.')
        parser.add_argument('--ssl_algo', type=str, default='supcontrast',
                            help='SSL training algorithm')
        parser.add_argument('--freeze_backbone', action='store_true',
                            help='Freeze backbone after SSL training and train CL only using single linear layer')
        parser.add_argument('--multitask', action='store_true',
                            help='Use rotation as multitask')
        parser.add_argument('--empty_buffer', action='store_true',
                            help='Do not use any buffer for experience replay')
        parser.add_argument('--ce_weight', type=float, default=1,
                            help='multitask weight for cross entropy')
        parser.add_argument('--rot_weight', type=float, default=1,
                            help='multitask weight for rotation')
        parser.add_argument('--img_size', type=int, required=True,
                            help='Input image size')
        parser.add_argument('--ssl_train_percentage', type=float, default=0.6,
                            help='Percentage of training time for SSL. Rest is for multitasking')
        parser.add_argument('--eval_c', action='store_true',
                            help='Use trained model for evaluation on natural corruption datasets')
        args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    if args.model == 'mer': setattr(args, 'batch_size', 1)
    dataset = get_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    if isinstance(dataset, ContinualDataset):
        train(model, dataset, args)
    else:
        assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
        ctrain(args)


if __name__ == '__main__':
    main()
