import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import *
from char_map import char_map, rev_char_map

from config import BASIC_CONFIG

def generate_model(feat_dim, num_classes, device):
    model = Seq2Seq(
        dim_in=feat_dim,
        num_classes=num_classes,
        encoder_layers=4,
        embed_dim=512,
        encoder_hidden_dim=512,
        decoder_hidden_dim=512,
        context_size=512
    ).to(device)

    return model


class Seq2Seq(nn.Module):
    def __init__(self, dim_in, num_classes, encoder_layers, embed_dim, encoder_hidden_dim, decoder_hidden_dim, context_size):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(encoder_layers, dim_in, encoder_hidden_dim, context_size)
        self.decoder = Decoder(num_classes, embed_dim, decoder_hidden_dim, context_size)

    def forward(self, x, lengths, y=None, mode='train', tf_prob=0.9):
        key, value, encoder_len = self.encoder(x, lengths, mode=mode)
        predictions = self.decoder(key, value, encoder_len, y=y, mode=mode, tf_prob=tf_prob)
        return predictions


class Encoder(nn.Module):
    def __init__(self, num_layers, dim_in, hidden_size, dim_out):
        super(Encoder, self).__init__()

        self.dropout = LockedDropout(0.2)

        self.num_layers = num_layers
        self.layer_0 = nn.LSTM(
            input_size=dim_in,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True
        )
        for i in range(1, num_layers):
            lstm = nn.LSTM(
                input_size=hidden_size * 4,
                hidden_size=hidden_size,
                num_layers=1,
                bidirectional=True
            )
            self.add_module('layer_' + str(i), lstm)

        self.key_projection = nn.Sequential(
            nn.Linear(hidden_size * 2, dim_out),
            nn.LeakyReLU(0.2),
            nn.Linear(dim_out, dim_out, bias=False),
        )

        self.value_projection = nn.Sequential(
            nn.Linear(hidden_size * 2, dim_out),
            nn.LeakyReLU(0.2),
            nn.Linear(dim_out, dim_out, bias=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, seq_len, mode='train'):
        x = x.squeeze(1).permute((2, 0, 1))

        for i in range(self.num_layers):
            if mode == 'train':
                x = self.dropout(x)
            x = pack_padded_sequence(x, seq_len, enforce_sorted=False)
            x, _ = getattr(self, 'layer_' + str(i))(x)
            x, seq_len = pad_packed_sequence(x)

            if i < self.num_layers - 1:
                max_len, batch_size, feat_dim = x.shape
                x = x.transpose(0, 1)
                x = x.contiguous().view(batch_size, max_len // 2, feat_dim * 2)
                x = x.transpose(0, 1)
                seq_len = seq_len // 2

        key = self.key_projection(x)
        value = self.value_projection(x)
        return key, value, seq_len


class Decoder(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_dim, context_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim, padding_idx=char_map['<eos>'])
        self.lstm1 = nn.LSTMCell(input_size=embed_dim + context_size, hidden_size=hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=context_size)
        self.attention = Attention()

        self.dropout = 0.2
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.context_size = context_size
        feat_dim = context_size * 2
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
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

    def forward(self, key, value, seq_len, y=None, mode='train', tf_prob=0.9):
        key_seq_max_len, batch_size, key_value_size = key.shape

        if mode == 'train':
            max_len = y.shape[1]
            char_embeddings = self.embedding(y)
        else:
            max_len = 600
        predictions = []
        prediction = torch.zeros(batch_size, self.num_classes).cuda()
        hidden_states = [None, None, None]

        context = torch.zeros(batch_size, self.context_size).cuda()
        attentions = []

        for i in range(max_len):
            if mode == 'train':
                if i > 0 and random.random() < tf_prob:
                    char_embed = char_embeddings[:, i-1]
                else:
                    char_embed = self.embedding(prediction.argmax(dim=1))
            else:
                char_embed = self.embedding(prediction.argmax(dim=1))

            y_context = torch.cat([char_embed, context], dim=1)
            hidden_states[0] = self.lstm1(y_context, hidden_states[0])

            lstm2_hidden = hidden_states[0][0]
            hidden_states[1] = self.lstm2(lstm2_hidden, hidden_states[1])

            output = hidden_states[1][0]
            context = self.attention(output, key, value, seq_len, attentions)

            output_context = torch.cat([output, context], dim=1)
            prediction = self.fc(output_context)
            predictions.append(prediction.unsqueeze(1))

        if random.random() < 0.01:
            att = np.array(attentions)
            plot_attention(att)
        return torch.cat(predictions, dim=1)


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, seq_len=None, attentions=None):
        query = query.unsqueeze(1)
        key = key.permute(1, 2, 0)
        value = value.permute(1, 0, 2)

        energy = torch.bmm(query, key)
        if seq_len is not None:
            seq_mask = torch.arange(max(seq_len))[None, :] < seq_len[:, None]
            seq_mask = seq_mask.cuda()

            seq_mask = seq_mask.unsqueeze(1)
            energy = energy.masked_fill_(~seq_mask, float('-inf'))

        scores = F.softmax(energy, dim=2)
        attentions.append(scores[0][0].detach().cpu().numpy())
        context = torch.bmm(scores, value)
        return context.squeeze()


import matplotlib.pyplot as plt
import seaborn as sns
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