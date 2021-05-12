import os
import sys
import random
import datetime

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config import *
from metrics import Estimator
from train import train, evaluate
from data import generate_dataset
from modules import generate_model
from utils import print_config


def main():
    # create folder
    save_path = BASIC_CONFIG['save_path']
    if os.path.exists(save_path):
        overwirte = input('Save path {} exists.\nDo you want to overwrite it? (y/n)\n'.format(save_path))
        if overwirte != 'y':
            sys.exit(0)
    else:
        os.makedirs(save_path)

    # create logger
    record_path = BASIC_CONFIG['record_path']
    record_path = os.path.join(record_path, 'log-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = SummaryWriter(record_path)

    # print configuration
    print_config({
        'BASIC CONFIG': BASIC_CONFIG,
        'DATA CONFIG': DATA_CONFIG,
        'TRAIN CONFIG': TRAIN_CONFIG
    })

    # reproducibility
    seed = BASIC_CONFIG['random_seed']
    set_random_seed(seed)

    device = BASIC_CONFIG['device']
    num_classes = BASIC_CONFIG['num_classes']
    dimension = DATA_CONFIG['dimension']

    model = generate_model(dimension, num_classes, device)

    train_dataset, dev_dataset, test_dataset = generate_dataset(
        DATA_CONFIG['train_data'],
        DATA_CONFIG['train_label'],
        DATA_CONFIG['dev_data'],
        DATA_CONFIG['dev_label'],
        DATA_CONFIG['test_data'],
    )

    # create estimator and then train
    num_classes = BASIC_CONFIG['num_classes']
    estimator = Estimator(device)
    train(
        model=model,
        train_config=TRAIN_CONFIG,
        train_dataset=train_dataset,
        val_dataset=dev_dataset,
        save_path=save_path,
        estimator=estimator,
        device=device,
        logger=logger
    )

    # evaluate(model, './result/LeakyReLU/best_validation_weights.pt', test_dataset, estimator, 'cuda')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
