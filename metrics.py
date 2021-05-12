import torch
import numpy as np
from Levenshtein import distance
from ctcdecode import CTCBeamDecoder

from phoneme_list import PHONEME_MAP


class Estimator():
    def __init__(self, device='cpu'):
        self.device = device
        self.total_dist = 0
        self.num_samples = 0
        self.decoder = CTCBeamDecoder(
            PHONEME_MAP,
            beam_width=1,
            num_processes=16,
            blank_id=0,
            log_probs_input=False
        )

    def update(self, predictions, targets, input_lengths):
        predictions = predictions.transpose(0, 1)
        predictions = torch.softmax(predictions, dim=2)
        batch_size = predictions.shape[0]
        beam_results, beam_scores, timesteps, out_lens = self.decoder.decode(predictions, input_lengths)
        preds = [self.convert_to_string(beam_results[i][0], PHONEME_MAP, out_lens[i][0]) for i in range(batch_size)]
        gts = self.to_char(targets.data)

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
            sentence = "".join([PHONEME_MAP[idx.item()] for idx in d])
            sentences.append(sentence.strip())
        return sentences

    def convert_to_string(self, tokens, vocab, seq_len):
        return ''.join([vocab[x] for x in tokens[0:seq_len]])
