# Task Agnostic Representation Consolidation: A Self-supervised based Continual Learning Approach 
Official Repository for for CoLLAs 2022 paper "[Task Agnostic Representation Consolidation: A Self-supervised based Continual Learning Approach](https://arxiv.org/abs/2207.06267)"

This repo is built on top of the [Mammoth](https://github.com/aimagelab/mammoth) continual learning framework

## Setup

+ Use `python main.py` to run experiments.
+ Use argument `--load_best_args` to use the best hyperparameters for each of the evaluation setting from the paper.

 Examples:
    
    ## How to run?
    python --model er --dataset seq-cifar10 --img_size 32  --buffer_size 200 --load_best_args --train_ssl --ssl_train_percentage 0.9 --tensorboard --multitask --ce_weight 1 --rot_weight 1 --notes 'ER + TARC'

    python --model si --dataset rot-mnist --img_size 28 --load_best_args --train_ssl --ssl_train_percentage 0.6 --tensorboard  --multitask --ce_weight 1 --rot_weight 1 --notes 'SI + TARC'

    python --model ewc_on --dataset rot-mnist --img_size 28 --load_best_args --train_ssl --ssl_train_percentage 0.6 --tensorboard  --multitask --ce_weight 1 --rot_weight 1  --notes 'oEWC + TARC'
 
    + For multi-objective learning:
    --multitask --ce_weight 1 --rot_weight 1
 
    + For task-agnostic learning:
    --train_ssl --ssl_train_percentage 0.9
    
    

**Class-Il / Task-IL settings**

+ Sequential CIFAR-10
+ Sequential CIFAR-100
+ Sequential Tiny ImageNet
+ Sequential STL-10

**Domain-IL settings**

+ Rotated MNIST

**General Continual Learning setting**

+ MNIST-360

## Requirements

- torch==1.7.0

- torchvision==0.9.0 

- quadprog==0.1.7


## Cite Our Work

If you find the code useful in your research, please consider citing our paper:



