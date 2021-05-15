import random
import torch
import numpy as np
import torch.nn as nn
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.nn.utils.rnn import *
from char_map import char_map, rev_char_map

from config import BASIC_CONFIG

def generate_model(feat_dim, num_classes, device):
    model = Seq2Seq(
        dim_in=feat_dim,
        num_classes=num_classes,
        encoder_layers=3,
        embed_dim=256,
        encoder_hidden_dim=256,
        decoder_hidden_dim=256
    ).to(device)

    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


class Seq2Seq(nn.Module):
    def __init__(self, dim_in, num_classes, encoder_layers, embed_dim, encoder_hidden_dim, decoder_hidden_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(encoder_layers, dim_in, encoder_hidden_dim)
        self.decoder = Decoder(num_classes, embed_dim, decoder_hidden_dim, encoder_hidden_dim * 2)

    def forward(self, x, lengths, y=None, mode='train', tf_prob=0.9):
        feats, encoder_len = self.encoder(x, lengths, mode=mode)
        predictions = self.decoder(feats, encoder_len, y=y, mode=mode, tf_prob=tf_prob)
        return predictions


class Encoder(nn.Module):
    def __init__(self, num_layers, dim_in, hidden_size):
        super(Encoder, self).__init__()

        # self.dropout = LockedDropout(0.2)
        self.gru = nn.GRU(
            input_size=dim_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, seq_len, mode='train'):
        x = x.squeeze(1).permute((2, 0, 1))
        packed_x = pack_padded_sequence(x, seq_len, enforce_sorted=False)
        packed_out, _ = self.gru(packed_x)
        out, out_len = pad_packed_sequence(packed_out)
        return out, out_len


class Decoder(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_dim, feat_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim, padding_idx=char_map['blank'])
        self.gru = nn.GRUCell(input_size=embed_dim, hidden_size=hidden_dim)
        self.attention = ContentBasedAttention(hidden_dim, feat_dim, 512)

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.context_dim = 512 + hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(self.context_dim, embed_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(embed_dim, num_classes, bias=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # weight tying
        self.fc[-1].weight = self.embedding.weight

    def forward(self, feats, seq_len, y=None, mode='train', tf_prob=0.9):
        feat_seq_len, batch_size, _ = feats.shape

        if mode == 'train':
            max_len = y.shape[1]
            char_embeddings = self.embedding(y)
        else:
            max_len = 300

        predictions = []
        alpha = torch.ones(feat_seq_len, batch_size, 1).cuda() / feat_seq_len
        prediction = torch.zeros(batch_size, self.num_classes).cuda()
        hidden_state = torch.zeros(batch_size, self.hidden_dim).cuda()

        attentions = []

        for i in range(max_len):
            if mode == 'train':
                if i > 0 and random.random() < tf_prob:
                    char_embed = char_embeddings[:, i-1]
                else:
                    char_embed = self.embedding(prediction.argmax(dim=1))
            else:
                char_embed = self.embedding(prediction.argmax(dim=1))

            glimpse, alpha = self.attention(hidden_state, feats, alpha, seq_len)
            context = torch.cat([hidden_state, glimpse], dim=1)
            prediction = self.fc(context)
            predictions.append(prediction.unsqueeze(1))

            hidden_state = self.gru(char_embed, hidden_state)

            attentions.append(alpha[:, 0].detach().cpu().numpy())

        if random.random() < 0.01:
            att = np.array(attentions).squeeze()
            plot_attention(att)
        return torch.cat(predictions, dim=1)


class ContentBasedAttention(nn.Module):
    def __init__(self, hid_dim, feat_dim, en_dim):
        super(ContentBasedAttention, self).__init__()
        self.F = nn.Conv1d(1, 201, kernel_size=9, padding=4, bias=False)
        self.W = nn.Linear(hid_dim, en_dim, bias=True)
        self.V = nn.Linear(feat_dim, en_dim, bias=True)
        self.U = nn.Linear(201, en_dim, bias=True)
        self.w = nn.Linear(en_dim, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, sp, hs, alpha, seq_len=None):

        sp = self.W(sp)
        hs = self.V(hs)

        alpha = alpha.permute(1, 2, 0)
        f = self.F(alpha).permute(2, 0, 1)
        f = self.U(f)

        c = self.tanh(sp + hs + f)
        energy = self.w(c)

        if seq_len is not None:
            seq_mask = torch.arange(max(seq_len))[None, :] < seq_len[:, None]
            seq_mask = seq_mask.cuda()

            seq_mask = seq_mask.transpose(0, 1)
            seq_mask = seq_mask.unsqueeze(2)
            energy = energy.masked_fill_(~seq_mask, float('-inf'))

        alpha = F.softmax(energy, dim=0)
        glimpse = (alpha * hs).sum(0)
        return glimpse, alpha


def plot_attention(attention):
    plt.clf()
    sns.heatmap(attention, cmap='GnBu')
    plt.savefig('{}/attn.png'.format(BASIC_CONFIG['save_path']))


class WarmupLRScheduler():
    def __init__(self, optimizer, warmup_epochs, initial_lr):
        self.epoch = 0
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr

    def step(self):
        if self.epoch <= self.warmup_epochs:
            self.epoch += 1
            curr_lr = (self.epoch / self.warmup_epochs) * self.initial_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = curr_lr

    def is_finish(self):
        return self.epoch >= self.warmup_epochs


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


# from torchnlp
class LockedDropout(nn.Module):
    def __init__(self, p=0.5):
        self.p = p
        super().__init__()

    def forward(self, x):
        x = x.clone()
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        return x * mask


# https://github.com/paniabhisek/maxout/blob/master/maxout.py
class MaxoutMLP(nn.Module):
    """Maxout using Multilayer Perceptron"""

    def __init__(self, input_size,
                 linear_layers, linear_neurons):
        super(MaxoutMLP, self).__init__()

        # initialize variables
        self.input_size = input_size
        self.linear_layers = linear_layers
        self.linear_neurons = linear_neurons

        # batch normalization layer
        self.BN = torch.nn.BatchNorm1d(self.linear_neurons)

        # pytorch not able to reach the parameters of
        # linear layer inside a list
        self.params = torch.nn.ParameterList()
        self.z = []
        for layer in range(self.linear_layers):
            self.z.append(torch.nn.Linear(self.input_size,
                                          self.linear_neurons))
            self.params.extend(list(self.z[layer].parameters()))

    def forward(self, input_, is_norm=False, **kargs):
        h = None
        for layer in range(self.linear_layers):
            z = self.z[layer](input_)
            # norm + norm constraint
            if is_norm:
                z = self.BN(z)
                z = torch.where(z <= kargs['norm_constraint'],
                                z, kargs['norm_upper'])
            if layer == 0:
                h = z
            else:
                h = torch.max(h, z)
        return h


def smooth_softmax(x, dim):
    maxes = torch.max(x, dim, keepdim=True)[0]
    x_sig = torch.sigmoid(x - maxes)
    x_sig_sum = torch.sum(x_sig, dim, keepdim=True)
    out = x_sig / x_sig_sum
    return out
