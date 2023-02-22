from torch import nn, pdist
import torch
import torch.nn.functional as F
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.utils.weight_norm import weight_norm
from torch.utils.data import Dataset

import numpy as np
import torch
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

from torch import nn, pdist
import torch.nn.functional as F
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.utils.weight_norm import weight_norm
from torch.utils.data import Dataset
from transformers.utils.dummy_pt_objects import MODEL_MAPPING
from dataclasses import asdict

from .base_models import  PairwiseContactPredictionHead
from protein_models import ProteinBertModel, ProteinLSTMModel, ProteinResNetModel
from protein_models import ProteinLSTMConfig, ProteinBertConfig, ProteinResNetConfig
from transformers import get_linear_schedule_with_warmup

encoder_mapping = {
    'bert': ProteinBertModel,
    'lstm': ProteinLSTMModel,
    'resnet': ProteinResNetModel
}

config_mapping = {
    'bert': ProteinBertConfig,
    'lstm': ProteinLSTMConfig,
    'resnet': ProteinResNetConfig
}

def load_adam_optimizer_and_scheduler(model, args, train_dataset):
    optimizer = torch.optim.Adam(model.parameters())

    total_steps = len(
        train_dataset) // args.train_batch_size // args.gradient_accumulation_steps * args.num_train_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    return optimizer, scheduler

class DeepLearningforSequenceClassification(nn.Module):
    def __init__(self, model_args, num_labels, mean_output):
        super().__init__()
        self.model_args = model_args
        config_fn = config_mapping[model_args.model_type]
        encoder_fn = encoder_mapping[model_args.model_type]
        config = config_fn(**asdict(model_args))
        self.encoder = encoder_fn(config)
        
        self.config = config
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.num_labels= num_labels
        self.mean_output = mean_output
        self.problem_type=None
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        **kwargs
    ):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
        )

        if self.mean_output is not True:
            outputs_ = outputs[1]
        else:
            outputs_ = outputs
            attention_mask = attention_mask.bool()
            num_batch_size = attention_mask.size(0)
            outputs_ = torch.stack([outputs_[0][i, attention_mask[i, :], :].mean(dim=0) for i in
                                      range(num_batch_size)], dim=0)
        
        outputs_ = self.dropout(outputs_)
        logits = self.classifier(outputs_)
        
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
                
                
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
        
class DeepLearningforTokenClassification(nn.Module):
    def __init__(self, model_args, num_labels, mean_output):
        super().__init__()
        self.model_args = model_args
        config_fn = config_mapping[model_args.model_type]
        encoder_fn = encoder_mapping[model_args.model_type]
        config = config_fn(**asdict(model_args))
        self.encoder = encoder_fn(config)
        
        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.num_labels= num_labels
        self.mean_output = mean_output
        self.problem_type=None
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        protein_length = None,
        **kwargs
    ):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]
        
        if protein_length is not None:
            len = protein_length.tolist()[0]  # for retrieval only support batch_size = 1
            #print(sequence_output.shape)
            sequence_output = sequence_output[:,:len+2,:]   # [cls]x[sep] embedding
            
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None and protein_length is None:  # if have protein length, preprocessed before
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class DeepLearningforContactPrediction(nn.Module):
    def __init__(self, model_args, num_labels, mean_output):
        super().__init__()
        self.model_args = model_args
        config_fn = config_mapping[model_args.model_type]
        encoder_fn = encoder_mapping[model_args.model_type]
        config = config_fn(**asdict(model_args))
        self.encoder = encoder_fn(config)
        
        self.config = config
        self.predict = PairwiseContactPredictionHead(config.hidden_size, ignore_index=-1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.mean_output = mean_output
        self.num_labels = num_labels
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        protein_length = None,
        **kwargs
    ):  
        targets = labels
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]
        if protein_length is not None and len(protein_length)==1:
            length = protein_length.tolist()[0]  # for retrieval only support batch_size = 1
            sequence_output = sequence_output[:,:length+2,:]   
        # print(sequence_output.shape)
        output_precision = self.predict(sequence_output, protein_length, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        outputs = TokenClassifierOutput(hidden_states=sequence_output, attentions=None)
        outputs['loss'] = output_precision[0][0]
        outputs['logits'] = output_precision[1]
        outputs['prediction_score'] = output_precision[0][1]
        return outputs
    

class DeepLearningforInteractionPrediction(nn.Module):    
    def __init__(self, model_args, num_labels, mean_output):
        super().__init__()
        self.model_args = model_args
        config_fn = config_mapping[model_args.model_type]
        encoder_fn = encoder_mapping[model_args.model_type]
        config = config_fn(**asdict(model_args))
        self.encoder = encoder_fn(config)
        
        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, num_labels)
        self.num_labels= num_labels
        self.mean_output = mean_output
        self.problem_type= None

    def forward(
        self,
        input_ids_1=None,
        attention_mask_1=None,
        input_ids_2=None,
        attention_mask_2=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        protein_length_1=None,
        protein_length_2=None,
        **kwargs
    ):
        outputs_1 = self.encoder(
            input_ids_1,
            attention_mask=attention_mask_1,
        )
        if self.mean_output is not True:
            outputs_1_ = outputs_1[1]
        else:
            outputs_1_ = outputs_1
            attention_mask_1 = attention_mask_1.bool()
            num_batch_size_1 = attention_mask_1.size(0)
            outputs_1_ = torch.stack([outputs_1_[0][i, attention_mask_1[i, :], :].mean(dim=0) for i in
                                      range(num_batch_size_1)], dim=0)
        
        outputs_1_ = self.dropout(outputs_1_)
            
        outputs_2 = self.encoder(
            input_ids_2,
            attention_mask=attention_mask_2,
        )
        if self.mean_output is not True:
            outputs_2_ = outputs_2[1]
        else:
            outputs_2_ = outputs_2
            attention_mask_2 = attention_mask_2.bool()
            num_batch_size_2 = attention_mask_2.size(0)
            outputs_2_ = torch.stack([outputs_2_[0][i, attention_mask_2[i, :], :].mean(dim=0) for i in
                                        range(num_batch_size_2)], dim=0)
        
        outputs_2_ = self.dropout(outputs_2_)
        
        logits = self.classifier(torch.cat([outputs_1_, outputs_2_], dim=-1))
        
        loss = None
        
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
                
                
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
model_mapping = {
    'bert':
        {
        'remote_homology': DeepLearningforSequenceClassification,
        'contact': DeepLearningforContactPrediction,
        'fluorescence': DeepLearningforSequenceClassification,
        'stability': DeepLearningforSequenceClassification,
        'ss3': DeepLearningforTokenClassification,
        'ss8': DeepLearningforTokenClassification,
        'subcellular': DeepLearningforSequenceClassification,
        'ppi': DeepLearningforInteractionPrediction
        },
    'lstm':
        {
        'remote_homology': DeepLearningforSequenceClassification,
        'contact': DeepLearningforContactPrediction,
        'fluorescence': DeepLearningforSequenceClassification,
        'stability': DeepLearningforSequenceClassification,
        'ss3': DeepLearningforTokenClassification,
        'ss8': DeepLearningforTokenClassification,
        'subcellular': DeepLearningforSequenceClassification,
        'ppi': DeepLearningforInteractionPrediction
        },
    'resnet':
        {
        'remote_homology': DeepLearningforSequenceClassification,
        'contact': DeepLearningforContactPrediction,
        'fluorescence': DeepLearningforSequenceClassification,
        'stability': DeepLearningforSequenceClassification,
        'ss3': DeepLearningforTokenClassification,
        'ss8': DeepLearningforTokenClassification,
        'subcellular': DeepLearningforSequenceClassification,
        'ppi': DeepLearningforInteractionPrediction
        },
}