import esm_mine
#from tape.models.modeling_utils import PairwiseContactPredictionHead
from torch import nn, pdist
import torch
import torch.nn.functional as F
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.utils.weight_norm import weight_norm
from torch.utils.data import Dataset
from transformers import BertPreTrainedModel, BertModel, ESMPreTrainedModel, ESMModel, AdamW, \
    get_linear_schedule_with_warmup

import numpy as np
import torch
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput
from benchmark.base_models import PairwiseContactPredictionHead



class MSATokenizer:
    def __init__(self,):
        msa_transformer, msa_transformer_alphabet = esm_mine.pretrained.esm_msa1b_t12_100M_UR50S()
        # msa_transformer = msa_transformer.eval().cuda()
        msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()
        self.msa_transformer_batch_converter=msa_transformer_batch_converter

    def __call__(self, inputs,append_eos=False):
        res= self.msa_transformer_batch_converter(inputs)
        if append_eos:
            shape=res[2].shape
            end_pad=torch.ones(shape[0:-1]).unsqueeze(-1)
            res_new=torch.cat([res[2],end_pad],-1).long()
            res=(res[0],res[1],res_new)
        return res


    @classmethod
    def from_pretrained(cls,*tuples,**kwargs):

        return cls()



class MSAForTokenClassification(nn.Module):
    def __init__(self, num_labels,mean_output):
        super().__init__()
        msa_transformer, msa_transformer_alphabet = esm_mine.pretrained.esm_msa1b_t12_100M_UR50S()
        self.encoder =msa_transformer
        self.classifier = nn.Linear(msa_transformer.args.embed_dim, num_labels)
        self.num_labels= num_labels
        self.mean_output = mean_output


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
            protein_length=None,
            **kwargs
    ):


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids,
            repr_layers=[self.encoder.num_layers]
        )
        outputs=outputs['representations'][self.encoder.num_layers]

        sequence_output = outputs[:,0,:,:]

        if protein_length is not None:
            len = protein_length.tolist()[0]  # for retrieval only support batch_size = 1
            sequence_output = sequence_output[:, :len + 2, :]

        # sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
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
            # hidden_states=outputs,
            attentions=None,
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path,num_labels,mean_output):
        return cls(num_labels,mean_output)



class MSAForSequenceClassification(nn.Module):
    def __init__(self, num_labels,mean_output):
        super().__init__()
        msa_transformer, msa_transformer_alphabet = esm_mine.pretrained.esm_msa1b_t12_100M_UR50S()
        self.encoder =msa_transformer
        self.classifier = nn.Linear(msa_transformer.args.embed_dim, num_labels)
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
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids,
            repr_layers=[self.encoder.num_layers]
        )
        outputs=outputs['representations'][self.encoder.num_layers]

        sequence_output = outputs[:,0,0,:]

        logits = self.classifier(sequence_output)



        loss = None
        if labels is not None:
            if self.problem_type is None:
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=outputs,
            attentions=None,
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path,num_labels,mean_output):
        return cls(num_labels,mean_output)




class MSAForProteinContactPrediction(nn.Module):
    def __init__(self, num_labels,mean_output):
        super().__init__()
        msa_transformer, msa_transformer_alphabet = esm_mine.pretrained.esm_msa1b_t12_100M_UR50S()
        self.encoder =msa_transformer
        self.classifier = nn.Linear(msa_transformer.args.ffn_embed_dim, num_labels)
        # self.classifier = SimpleMLP(config.hidden_size, 512, config.num_labels)
        self.predict = PairwiseContactPredictionHead(msa_transformer.args.embed_dim, ignore_index=-1)
        self.mean_output = mean_output
        # self.classifier = SimpleMLP(config.hidden_size, 512, config.num_labels)


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
            protein_length=None,
            **kwargs
    ):

        targets = labels
        outputs = self.encoder(
            input_ids,
            repr_layers=[self.encoder.num_layers]
        )
        outputs=outputs['representations'][self.encoder.num_layers]

        sequence_output = outputs[:,0,:,:]
        len = protein_length.tolist()[0]  # for retrieval only support batch_size = 1
        # assert sequence_output.shape[1]>=len+2
        sequence_output = sequence_output[:,:len+2,:]
        # print(sequence_output.shape)
        # try:
        #     assert sequence_output.shape[1]==targets.shape[1]+2
        # except:
        # print(input_ids,input_ids.shape)
        # print(protein_length)
        # print(sequence_output,sequence_output.shape)
        # print(targets,targets.shape)
        output_precision = self.predict(sequence_output, protein_length, targets)
        # (loss), prediction_scores, (hidden_states), (attentions)
        outputs_final={}
        outputs_final['hidden']=outputs
        outputs_final['loss'] = output_precision[0][0]
        outputs_final['logits'] = output_precision[1]
        outputs_final['prediction_score'] = output_precision[0][1]
        print(outputs_final['prediction_score'])
        return outputs_final

    @classmethod
    def from_pretrained(cls, model_name_or_path,num_labels,mean_output):
        return cls(num_labels,mean_output)



class MSAForInteractionPrediction(nn.Module):
    def __init__(self, num_labels,mean_output):
        super().__init__()
        msa_transformer, msa_transformer_alphabet = esm_mine.pretrained.esm_msa1b_t12_100M_UR50S()
        self.encoder =msa_transformer
        self.classifier = nn.Linear(msa_transformer.args.embed_dim* 2, num_labels)
        self.num_labels= num_labels
        self.mean_output = mean_output
        self.problem_type=None

    def forward(
        self,
        input_ids_1=None,
        attention_mask_1=None,
        input_ids_2=None,
        attention_mask_2=None,
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids_1,
            repr_layers=[self.encoder.num_layers]
        )
        outputs=outputs['representations'][self.encoder.num_layers]

        outputs_1_ = outputs[:,0,0,:]

        outputs = self.encoder(
            input_ids_2,
            repr_layers=[self.encoder.num_layers]
        )
        outputs=outputs['representations'][self.encoder.num_layers]

        outputs_2_ = outputs[:,0,0,:]

        # logits = self.classifier(sequence_output)
        logits = self.classifier(torch.cat([outputs_1_, outputs_2_], dim=-1))


        loss = None
        if labels is not None:
            if self.problem_type is None:
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            raise NotImplementedError()
            # output = (logits,) + outputs[2:]
            # return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=outputs,
            attentions=None,
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path,num_labels,mean_output):
        return cls(num_labels,mean_output)



