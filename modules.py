import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import *
from utils import one_hot


def generate_model(feat_dim, num_classes, device):
    model = LargeLSTMNet(
        feat_dim,
        num_classes,
    ).to(device)

    return model


class LargeLSTMNet(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super(LargeLSTMNet, self).__init__()

        hidden_size = 512
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=5,
            dropout=0.2,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x, lengths):
        x = x.squeeze(1).permute((2, 0, 1))
        packed_x = pack_padded_sequence(x, lengths, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, out_len = pad_packed_sequence(packed_out)
        out = self.fc(out)
        return out, out_len


class ClippedCosineAnnealingLR():
    def __init__(self, optimizer, T_max, min_lr):
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        self.min_lr = min_lr
        self.finish = False

    def step(self):
        if not self.finish:
            self.scheduler.step()
            curr_lr = self.optimizer.param_groups[0]['lr']
            if curr_lr < self.min_lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.min_lr
                self.finish = True

    def is_finish(self):
        return self.finish
