import torch
import numpy as np
from Levenshtein import distance
from torch.types import Number
from char_map import CHAR_LIST, char_map


class Estimator():
    def __init__(self, device='cpu'):
        self.device = device
        self.total_dist = 0
        self.num_samples = 0

    def update(self, predictions, targets):
        predictions = torch.softmax(predictions, dim=2)
        batch_size = predictions.shape[0]

        input_lengths = self.length_to_eos(predictions)
        results = self.greedy_search(predictions, input_lengths)
        preds = [self.convert_to_string(results[i], CHAR_LIST) for i in range(batch_size)]
        print('======================')
        print(preds[0])
        gts = [sent.replace('<sos>', '').replace('<eos>', '') for sent in self.to_char(targets.data)]
        print(gts[0])

        # update metrics
        self.num_samples += batch_size
        self.total_dist += sum([distance(preds[i], gts[i]) for i in range(batch_size)])

    def get_dist(self, digits=-1):
        dist = self.total_dist / self.num_samples
        dist = dist if digits == -1 else round(dist, digits)
        return dist

    def reset(self):
        self.total_dist = 0
        self.num_samples = 0

    def to_char(self, data):
        sentences = []
        for d in data:
            sentence = "".join([CHAR_LIST[idx.item()] for idx in d])
            sentences.append(sentence.strip())
        return sentences

    def convert_to_string(self, tokens, vocab):
        return ''.join([vocab[x] for x in tokens])

    def length_to_eos(self, predictions):
        eos_id = char_map['<eos>']

        num_classes = predictions.shape[-1]
        eos = torch.eye(num_classes)[eos_id].cuda()
        predictions[:, -1] = eos
        pred = predictions.argmax(dim=2)
        length = torch.max(pred == eos_id, dim=1)[1]

        assert length.shape[0] == predictions.shape[0]
        return length

    def greedy_search(self, predictions, input_lengths):
        eos_id = char_map['<eos>']

        results = []
        batch_size = predictions.shape[0]

        predictions = torch.argmax(predictions, dim=2)
        for batch in range(batch_size):
            preds = predictions[batch]
            in_len = input_lengths[batch]
            prev = None
            out_len = 0
            result = []
            for pred in preds:
                out_len += 1
                if pred == eos_id or out_len > in_len:
                    break

                if pred == prev:
                    continue
                else:
                    result.append(pred)
                    prev = pred
            
            results.append(result)

        return results