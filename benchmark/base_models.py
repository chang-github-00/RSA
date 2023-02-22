import torch
from torch import nn
import torch.nn.functional as F
class PairwiseContactPredictionHead(nn.Module):

    def __init__(self, hidden_size: int, ignore_index=-100):
        super().__init__()
        self.predict = nn.Sequential(
            nn.Dropout(), nn.Linear(2 * hidden_size, 2))
        self._ignore_index = ignore_index

    def forward(self, inputs, sequence_lengths, targets=None, **kwargs):
        prod = inputs[:, :, None, :] * inputs[:, None, :, :]
        diff = inputs[:, :, None, :] - inputs[:, None, :, :]
        pairwise_features = torch.cat((prod, diff), -1)
        prediction = self.predict(pairwise_features)
        prediction = (prediction + prediction.transpose(1, 2)) / 2
        prediction = prediction[:, 1:-1, 1:-1].contiguous()  # remove start/stop tokens
        outputs = (prediction,)

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            contact_loss = loss_fct(
                prediction.view(-1, 2), targets.view(-1))
            metrics = {'precision_at_l5': self.compute_precision_at_l5(sequence_lengths, prediction, targets),
                       'precision_at_l2': self.compute_precision_at_l2(sequence_lengths, prediction, targets),
                       'precision_at_l': self.compute_precision_at_l(sequence_lengths, prediction, targets)}
            loss_and_metrics = (contact_loss, metrics)
            outputs = (loss_and_metrics,) + outputs

        return outputs

    def compute_precision_at_l5(self, sequence_lengths, prediction, labels):
        with torch.no_grad():
            valid_mask = labels != self._ignore_index
            seqpos = torch.arange(valid_mask.size(1), device=sequence_lengths.device)
            x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
            valid_mask &= ((y_ind - x_ind) >= 6).unsqueeze(0)
            probs = F.softmax(prediction, 3)[:, :, :, 1]
            valid_mask = valid_mask.type_as(probs)
            correct = 0
            total = 0
            for length, prob, label, mask in zip(sequence_lengths, probs, labels, valid_mask):
                masked_prob = (prob * mask).view(-1)
                most_likely = masked_prob.topk(length // 5, sorted=False)
                selected = label.view(-1).gather(0, most_likely.indices)
                correct += selected.sum().float()
                total += selected.numel()
            return correct / total

    def compute_precision_at_l2(self, sequence_lengths, prediction, labels):
        with torch.no_grad():
            valid_mask = labels != self._ignore_index
            seqpos = torch.arange(valid_mask.size(1), device=sequence_lengths.device)
            x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
            valid_mask &= ((y_ind - x_ind) >= 6).unsqueeze(0)
            probs = F.softmax(prediction, 3)[:, :, :, 1]
            valid_mask = valid_mask.type_as(probs)
            correct = 0
            total = 0
            for length, prob, label, mask in zip(sequence_lengths, probs, labels, valid_mask):
                masked_prob = (prob * mask).view(-1)
                most_likely = masked_prob.topk(length // 2, sorted=False)
                selected = label.view(-1).gather(0, most_likely.indices)
                correct += selected.sum().float()
                total += selected.numel()
            return correct / total

    def compute_precision_at_l(self, sequence_lengths, prediction, labels):
        with torch.no_grad():
            valid_mask = labels != self._ignore_index
            seqpos = torch.arange(valid_mask.size(1), device=sequence_lengths.device)
            x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
            valid_mask &= ((y_ind - x_ind) >= 6).unsqueeze(0)
            probs = F.softmax(prediction, 3)[:, :, :, 1]
            valid_mask = valid_mask.type_as(probs)
            correct = 0
            total = 0
            for length, prob, label, mask in zip(sequence_lengths, probs, labels, valid_mask):
                masked_prob = (prob * mask).view(-1)
                most_likely = masked_prob.topk(length, sorted=False)
                selected = label.view(-1).gather(0, most_likely.indices)
                correct += selected.sum().float()
                total += selected.numel()
            return correct / total

