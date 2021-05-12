from functools import reduce
import os
from numpy.core.numeric import Inf
from numpy.lib.function_base import append

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from modules import *
from data import MelspectrogramsDataset, InferenceMelspectrogramsDataset
from utils import save_weights, print_msg


def train(model, train_config, train_dataset, val_dataset, save_path, estimator, device, logger=None):
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = initialize_optimizer(train_config, model)
    lr_scheduler, warmup_scheduler = initialize_lr_scheduler(train_config, optimizer)
    train_loader, val_loader = initialize_dataloader(train_config, train_dataset, val_dataset)

    # start training
    model.train()
    min_indicator = 999
    avg_loss, avg_dist = 0, 0
    tf_prob = 0.9
    for epoch in range(train_config['epochs']):
        # warmup scheduler update
        if warmup_scheduler and not warmup_scheduler.is_finish():
            warmup_scheduler.step()

        if epoch > 50:
            tf_prob = 0.9 - 0.3 * ((epoch - 50) / (train_config['epochs'] - 50))

        epoch_loss = 0
        progress = tqdm(enumerate(train_loader))
        for step, train_data in progress:
            spectrograms, labels, seq_lengths, gt_lengths = train_data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            # mask
            gt_mask = torch.arange(max(gt_lengths))[None, :] < gt_lengths[:, None]
            gt_mask = gt_mask.to(device)

            # forward
            y_pred = model(spectrograms, seq_lengths, y=labels, tf_prob=tf_prob)

            batch_size, seq_len, num_classes = y_pred.shape
            y_pred = y_pred.view(batch_size * seq_len, num_classes)
            labels = labels.view(batch_size * seq_len)
            gt_mask = gt_mask.view(batch_size * seq_len)

            loss = criterion(y_pred, labels)
            loss = (gt_mask * loss).sum() / batch_size

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (step + 1)

            progress.set_description(
                'epoch: [{} / {}], loss: {:.6f}'
                .format(epoch, train_config['epochs'], avg_loss)
            )

        # validation performance
        if (epoch + 1) % train_config['eval_interval'] == 0:
            eval(model, val_loader, estimator, device)
            dist = estimator.get_dist(6)
            print('validation distance: {}'.format(dist))
            if logger:
                logger.add_scalar('validation distance', dist, epoch)

            # save model
            indicator = dist
            if indicator < min_indicator:
                save_weights(model, os.path.join(save_path, 'best_validation_weights.pt'))
                min_indicator = indicator
                print_msg('Best in validation set. Model save at {}'.format(save_path))

        if (epoch + 1) % train_config['save_interval'] == 0:
            save_weights(model, os.path.join(save_path, 'epoch_{}.pt'.format(epoch)))

        # update learning rate
        curr_lr = optimizer.param_groups[0]['lr']
        if lr_scheduler and (not warmup_scheduler or warmup_scheduler.is_finish()):
            if train_config['lr_scheduler'] == 'reduce_on_plateau':
                lr_scheduler.step(avg_loss)
            else:
                lr_scheduler.step()

        # record
        if logger:
            logger.add_scalar('training loss', avg_loss, epoch)
            logger.add_scalar('training distance', avg_dist, epoch)
            logger.add_scalar('learning rate', curr_lr, epoch)

    # save final model
    save_weights(model, os.path.join(save_path, 'final_weights.pt'))
    if logger:
        logger.close()


def evaluate(model, checkpoint, test_dataset, estimator, device):
    weights = torch.load(checkpoint)
    model.load_state_dict(weights, strict=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=16,
        drop_last=False,
        pin_memory=True,
        collate_fn=InferenceMelspectrogramsDataset.collate_fn
    )

    print('Running on Test set...')
    inference_eval(model, test_loader, estimator, device)

    print('========================================')
    print('Finished! test distance: {}'.format(estimator.get_dist(6)))
    print('========================================')


def eval(model, dataloader, estimator, device):
    model.eval()
    torch.set_grad_enabled(False)

    estimator.reset()
    for test_data in dataloader:
        spectrograms, labels, seq_lengths, gt_lengths = test_data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        # forward
        y_pred = model(spectrograms, seq_lengths, mode='test')

        estimator.update(y_pred, labels)

    model.train()
    torch.set_grad_enabled(True)


def inference_eval(model, dataloader, estimator, device):
    from char_map import CHAR_LIST

    model.eval()
    torch.set_grad_enabled(False)

    estimator.reset()
    preds = []
    for test_data in tqdm(dataloader):
        spectrograms, seq_lengths = test_data
        spectrograms = spectrograms.to(device)

        # forward
        y_pred = model(spectrograms, seq_lengths, mode='test')

        y_pred = torch.softmax(y_pred, dim=2)
        batch_size = y_pred.shape[0]

        input_lengths = estimator.length_to_eos(y_pred)
        beam_results = estimator.greedy_search(y_pred, input_lengths)
        preds += [estimator.convert_to_string(beam_results[i], CHAR_LIST) for i in range(batch_size)]

    submission = open('./submission.csv', 'w')
    submission.write('id,label\n')
    for i, pred in enumerate(preds):
        submission.write('{},{}\n'.format(i, pred))

    submission.close()
    model.train()
    torch.set_grad_enabled(True)


# define data loader
def initialize_dataloader(train_config, train_dataset, val_dataset):
    batch_size = train_config['batch_size']
    num_workers = train_config['num_workers']
    pin_memory = train_config['pin_memory']

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        collate_fn=MelspectrogramsDataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        collate_fn=MelspectrogramsDataset.collate_fn
    )

    return train_loader, val_loader


# define optmizer
def initialize_optimizer(train_config, model):
    optimizer_strategy = train_config['optimizer']
    learning_rate = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    momentum = train_config['momentum']
    nesterov = train_config['nesterov']
    if optimizer_strategy == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )
    elif optimizer_strategy == 'ADAM':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise NotImplementedError('Not implemented optimizer.')

    return optimizer


# define learning rate scheduler
def initialize_lr_scheduler(train_config, optimizer):
    learning_rate = train_config['learning_rate']
    warmup_epochs = train_config['warmup_epochs']
    scheduler_strategy = train_config['lr_scheduler']
    scheduler_config = train_config['lr_scheduler_config']

    if scheduler_strategy == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
    elif scheduler_strategy == 'multiple_steps':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_config)
    elif scheduler_strategy == 'reduce_on_plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config)
    elif scheduler_strategy == 'exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_config)
    elif scheduler_strategy == 'clipped_cosine':
        lr_scheduler = ClippedCosineAnnealingLR(optimizer, **scheduler_config)
    else:
        raise NotImplementedError()

    if warmup_epochs > 0:
        warmup_scheduler = WarmupLRScheduler(optimizer, warmup_epochs, learning_rate)
    else:
        warmup_scheduler = None

    return lr_scheduler, warmup_scheduler
