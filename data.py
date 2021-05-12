import math
import random
import torch
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset

from utils import mean_and_std
from char_map import char_map, rev_char_map


def generate_dataset(train_data, train_label, dev_data, dev_label, test_data):
    train_data = np.load(train_data, allow_pickle=True)
    train_label = np.load(train_label, allow_pickle=True)
    dev_data = np.load(dev_data, allow_pickle=True)
    dev_label = np.load(dev_label, allow_pickle=True)
    test_data = np.load(test_data, allow_pickle=True)

    mean, std = mean_and_std(train_data)
    train_preprocess = Preprocess(mean, std, training=False)
    test_preprocess = Preprocess(mean, std, training=False)

    train_dataset = MelspectrogramsDataset(train_data, train_label, transform=train_preprocess)
    dev_dataset = MelspectrogramsDataset(dev_data, dev_label, transform=test_preprocess)
    test_dataset = InferenceMelspectrogramsDataset(test_data, transform=test_preprocess)
    return train_dataset, dev_dataset, test_dataset


class MelspectrogramsDataset(Dataset):
    def __init__(self, np_data, np_label, transform=None):
        super(MelspectrogramsDataset, self).__init__()
        self.data = [torch.FloatTensor(sample) for sample in np_data]
        self.label = self.label_map(np_label)
        self.transform = transform

    def __getitem__(self, idx):
        utterance = self.data[idx]
        phoneme = self.label[idx]
        input_length = utterance.shape[0]
        phoneme_length = phoneme.shape[0]

        if self.transform:
            utterance = self.transform(utterance)
        return utterance, phoneme, input_length, phoneme_length

    def __len__(self):
        return len(self.data)

    def collate_fn(batch):
        max_len = max([b[2] for b in batch])
        pad_len = math.ceil(max_len / 8) * 8
        temp_tensor = torch.zeros(pad_len, 40)

        spectrograms = rnn_utils.pad_sequence([b[0] for b in batch] + [temp_tensor], batch_first=True)[:-1].unsqueeze(1).transpose(2, 3)
        labels = rnn_utils.pad_sequence([b[1] for b in batch], batch_first=True)
        input_lengths = torch.as_tensor([b[2] for b in batch])
        phoneme_lengths = torch.as_tensor([b[3] for b in batch])

        input_lengths = torch.ceil(input_lengths / 8) * 8
        return spectrograms, labels, input_lengths, phoneme_lengths

    def label_map(self, np_label):
        labels = []
        for words in np_label:
            chars = ' '.join([word.decode('UTF-8') for word in words])
            label = [char_map[char] for char in chars]
            # label = [char_map[char] for char in words[0]]
            label.append(char_map['<eos>'])
            labels.append(torch.as_tensor(label))

        return labels


class InferenceMelspectrogramsDataset(Dataset):
    def __init__(self, np_data, transform=None):
        super(InferenceMelspectrogramsDataset, self).__init__()
        self.data = [torch.FloatTensor(sample) for sample in np_data]
        self.transform = transform

    def __getitem__(self, idx):
        utterance = self.data[idx]
        input_length = utterance.shape[0]

        if self.transform:
            utterance = self.transform(utterance)
        return utterance, input_length

    def __len__(self):
        return len(self.data)

    def collate_fn(batch):
        max_len = max([b[1] for b in batch])
        pad_len = math.ceil(max_len / 8) * 8
        temp_tensor = torch.zeros(pad_len, 40)

        spectrograms = rnn_utils.pad_sequence([b[0] for b in batch] + [temp_tensor], batch_first=True)[:-1].unsqueeze(1).transpose(2, 3)
        input_lengths = torch.as_tensor([b[1] for b in batch])

        input_lengths = torch.ceil(input_lengths / 8) * 8
        return spectrograms, input_lengths


class Preprocess():
    def __init__(self, mean, std, training=False):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)
        self.training = training

    def __call__(self, x):
        x = (x - self.mean) / self.std
        return x
