import torch.nn as nn
from typing import Tuple, Optional, Union, Dict, Any, List
import torch
from transformers.modeling_outputs import ModelOutput
import torch.nn.functional as F


class MSAAugmentedProteinModel(nn.Module):
    def __init__(self, data_config, model_config, retrieval_config, model, tokenizer):
        super().__init__()
        self.model_type = model_config.model_type
        self.max_len = retrieval_config.concat_max_len
        self.base_model = model
        self.task_name = data_config.task_name
        self._ignore_index = -1
            
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


    # only use this under with torch.no_grad()
    def forward(self, **inputs):
        #new_input_ids, new_attention_masks, origin_lengths, probs = self.knn_retrieve(input_ids = inputs['input_ids'], features=inputs['features'], return_mask=True)
        
        new_input_ids = inputs['input_ids']
        new_attention_masks = inputs['attention_mask']
        origin_lengths = inputs['origin_length']
        
        b, k, l = new_input_ids.shape
        logits = None
        loss = None
        metrics = None
        probs = torch.ones([b, k], device = new_input_ids.device).float()/k
        
        for i in range(k):
            input_ids = new_input_ids[:,i,:]
            length = origin_lengths
            inputs['input_ids'] = input_ids
            inputs['origin_length'] = origin_lengths
            inputs['attention_mask'] = new_attention_masks[:, i, :]
            outputs = self.base_model(**inputs)
            
            
            if logits is None:
                logits = probs[0,i] * outputs['logits']
                loss = probs[0,i] * outputs['loss']
            else:
                logits += probs[0,i] * outputs['logits']
                loss += probs[0,i] * outputs['loss']

        if self.task_name == 'contact':
            metrics = {'precision_at_l5':self.compute_precision_at_l5(origin_lengths, logits, inputs["labels"]),
                       'precision_at_l2':self.compute_precision_at_l2(origin_lengths, logits, inputs["labels"]),
                       'precision_at_l':self.compute_precision_at_l(origin_lengths, logits, inputs["labels"])}
            
            outputs['loss'] = loss
            outputs['logits'] = logits
            outputs['prediction_score'] = metrics
            return outputs
        
        else:
            return ModelOutput(
                loss=loss,
                logits=logits,
                hidden_states=None,
                attentions=None,
            )
        