from .retriever import Retriever
import torch.nn as nn
from typing import Tuple, Optional, Union, Dict, Any, List
import torch
from transformers.modeling_outputs import ModelOutput
import torch.nn.functional as F
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss

class KNNProteinModel(nn.Module):  # this is the model for training, using concatenated sequences (query + retrieved sequences)
    def __init__(self, data_config, model_config, retrieval_config, model, tokenizer):
        super().__init__()
        self.retriever = Retriever(retrieval_config, tokenizer)
        self.model_type = model_config.model_type
        self.max_len = retrieval_config.concat_max_len
        self.base_model = model
        self.task_name = data_config.task_name
        self._ignore_index = -1
        self.num_labels = self.base_model.num_labels
        self.weight = retrieval_config.weight


    def knn_retrieve(self, input_ids, features, return_mask=False, weight="distance"):
        return self.retriever.retrieve_concat(features, input_ids, max_length=self.max_len, weight=self.weight, return_mask=return_mask)
    
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
        if self.task_name == 'ppi':
            new_input_ids_1, new_attention_masks_1, origin_lengths_1, probs_1 = self.knn_retrieve(input_ids = inputs['input_ids_1'], features=inputs['features_1'], weight=self.weight, return_mask=True)
            b_1, k_1, l_1 = new_input_ids_1.shape
        
            new_input_ids_2, new_attention_masks_2, origin_lengths_2, probs_2 = self.knn_retrieve(input_ids = inputs['input_ids_2'], features=inputs['features_2'], weight=self.weight, return_mask=True)
            b_2, k_2, l_2 = new_input_ids_2.shape

            logits = None
            loss = None
            metrics = None
            
            probs = (probs_1 + probs_2)/2
            
            for i in range(k_1):
                input_ids_1 = new_input_ids_1[:,i,:]
                length_1 = origin_lengths_1
                inputs['input_ids_1'] = input_ids_1
                inputs['protein_length_1'] = origin_lengths_1
                inputs['attention_mask_1'] = new_attention_masks_1[:,i,:]
                
                input_ids_2 = new_input_ids_2[:,i,:]
                length_2 = origin_lengths_2
                inputs['input_ids_2'] = input_ids_2
                inputs['protein_length_2'] = origin_lengths_2
                inputs['attention_mask_2'] = new_attention_masks_2[:,i,:]
                
                outputs = self.base_model(**inputs)
                
                if logits is None:
                    logits = probs[0,i] * outputs['logits']
                else:
                    logits += probs[0,i] * outputs['logits']
                    
                
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1,self.num_labels), inputs["labels"].view(-1))
                
                return ModelOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=None,
                    attentions=None,
                )
        else:
            new_input_ids, new_attention_masks, origin_lengths, probs = self.knn_retrieve(input_ids = inputs['input_ids'], features=inputs['features'], weight=self.weight, return_mask=True)
            b, k, l = new_input_ids.shape
            logits = None
            loss = None
            metrics = None
            
            for i in range(k):
                input_ids = new_input_ids[:,i,:]
                length = origin_lengths  # not accurate here, i will see how to fix
                inputs['input_ids'] = input_ids
                inputs['protein_length'] = origin_lengths
                inputs['attention_mask'] = new_attention_masks[:, i, :]
                outputs = self.base_model(**inputs)
                
                
                if logits is None:
                    logits = probs[0,i] * outputs['logits']
                    loss = probs[0,i] * outputs['loss']  # not accurate here, i will see how to fix
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
                
                
class KNNProteinModelParallel(nn.Module): # this is the model for training, using parallel input of both query and retrieved sequences
    def __init__(self, data_config, model_config, retrieval_config, model, tokenizer):
        super().__init__()
        self.retriever = Retriever(retrieval_config, tokenizer)
        self.model_type = model_config.model_type
        self.max_len = retrieval_config.concat_max_len  # in KNNProteinModelParallel, we use concat_max_len == original max_len
        self.base_model = model
        self.task_name = data_config.task_name
        self._ignore_index = -1
        self.num_labels = self.base_model.num_labels
        self.weight = retrieval_config.weight
        
        
    def knn_retrieve(self, input_ids, features, return_mask=False, weight="distance"):
        return self.retriever.retrieve(features, input_ids, max_length=self.max_len, return_mask=return_mask, weight=weight)
    
    def forward(self, **inputs): #this model is only for classification task and ppi task, not availabel for token level task
        if self.task_name == 'ppi':
            new_input_ids_1, new_attention_masks_1, lengths_1, probs_1 = self.knn_retrieve(input_ids = inputs['input_ids_1'], features=inputs['features_1'], weight=self.weight, return_mask=True)
            b_1, k_1, l_1 = new_input_ids_1.shape
            
            new_input_ids_2, new_attention_masks_2, lengths_2, probs_2 = self.knn_retrieve(input_ids = inputs['input_ids_2'], features=inputs['features_2'], weight=self.weight, return_mask=True)
            b_2, k_2, l_2 = new_input_ids_2.shape
            
            
            logits = None
            loss = None
            metrics = None
            
            probs = (probs_1 + probs_2)/2
            for i in range(k_1):
                input_ids_1 = new_input_ids_1[:,i,:]
                length_1 = lengths_1
                inputs['input_ids_1'] = input_ids_1
                inputs['protein_length_1'] = lengths_1
                inputs['attention_mask_1'] = new_attention_masks_1[:,i,:]
                
                input_ids_2 = new_input_ids_2[:,i,:]
                length_2 = lengths_2
                inputs['input_ids_2'] = input_ids_2
                inputs['protein_length_2'] = lengths_2
                inputs['attention_mask_2'] = new_attention_masks_2[:,i,:]
                
                outputs = self.base_model(**inputs)
                
                if logits is None:
                    logits = probs[0,i] * outputs['logits']
                else:
                    logits += probs[0,i] * outputs['logits']
            
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1,self.num_labels), inputs["labels"].view(-1))
                
                return ModelOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=None,
                    attentions=None,
                )
        else:
            new_input_ids, new_attention_masks, lengths, probs = self.knn_retrieve(input_ids = inputs['input_ids'], features=inputs['features'], weight=self.weight, return_mask=True)
            b, k, l = new_input_ids.shape
            logits = None 
            loss = None
            metrics = None
            
            for i in range(k):
                input_ids = new_input_ids[:,i,:]
                length = lengths
                inputs['input_ids'] = input_ids
                inputs['protein_length'] = lengths # not accurate here, i will see how to fix
                inputs['attention_mask'] = new_attention_masks[:, i, :]
                outputs = self.base_model(**inputs)
                    
                if logits is None:
                    logits = probs[0,i] * outputs['logits']
                    loss = probs[0,i] * outputs['loss']
                else:
                    logits += probs[0,i] * outputs['logits']
                    loss += probs[0,i] * outputs['loss']
                    
            
            return ModelOutput(
                        loss=loss,
                        logits=logits,
                        hidden_states=None,
                        attentions=None,
                    )