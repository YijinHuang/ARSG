import jiwer
import torch
import numpy as np
from Levenshtein import distance
from char_map import CHAR_LIST, char_map
from ctcdecode import CTCBeamDecoder

class Estimator():
    def __init__(self, device='cpu'):
        self.device = device
        self.preds = []
        self.gts = []
        self.decoder = CTCBeamDecoder(
            CHAR_LIST,
            beam_width=1,
            num_processes=16,
            blank_id=0,
            log_probs_input=False
        )

    def update(self, predictions, targets):
        predictions = torch.softmax(predictions, dim=2)
        batch_size = predictions.shape[0]

        input_lengths = self.length_to_eos(predictions)
        beam_results, beam_scores, timesteps, out_lens = self.decoder.decode(predictions, input_lengths)
        preds = [self.convert_to_string(beam_results[i][0], CHAR_LIST, out_lens[i][0]) for i in range(batch_size)]
        print('======================')
        print(preds[0])
        gts = [sent.replace('blank', '').replace('blank', '') for sent in self.to_char(targets.data)]
        print(gts[0])

        # update metrics
        self.preds += preds
        self.gts += gts

    def get_dist(self, digits=-1):
        # dist = self.total_dist / self.num_samples
        err = jiwer.wer(self.gts, self.preds)
        err = err if digits == -1 else round(err, digits)
        return err

    def reset(self):
        self.preds = []
        self.gts = []

    def to_char(self, data):
        sentences = []
        for d in data:
            sentence = " ".join([CHAR_LIST[idx.item()] for idx in d])
            sentences.append(sentence.strip())
        return sentences

    def convert_to_string(self, tokens, vocab, seq_len):
        return ' '.join([vocab[x] for x in tokens[0:seq_len]])

    def length_to_eos(self, predictions):
        eos_id = char_map['blank']

        num_classes = predictions.shape[-1]
        eos = torch.eye(num_classes)[eos_id].cuda()
        predictions[:, -1] = eos
        pred = predictions.argmax(dim=2)
        length = torch.max(pred == eos_id, dim=1)[1]

        assert length.shape[0] == predictions.shape[0]
        return length

    def greedy_search(self, predictions, input_lengths):
        eos_id = char_map['blank']

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
