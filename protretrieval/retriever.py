
from .knn import KNN_Dstore
import torch.nn.functional as F
import numpy as np
import torch
import re
import transformers
from transformers.utils import logging

logger = logging.get_logger(__name__)

class Retriever:
    def __init__(self, args, tokenizer):
        super().__init__()
        self.knn_dstore = KNN_Dstore(args)
        self.tokenizer = tokenizer
    
    def retrieve(self, query_features, query_input_ids, max_length=None, return_tensors="pt", return_mask=False, weight="uniform"):
        dists, re_indexes = self.knn_dstore.get_knns(query_features)
        dists = [0.0] + dists
        if weight == "uniform":
            probs = torch.ones_like(dists) / dists.shape[1]
        elif weight == "distance":
            dists = -1 * dists
            dists = torch.from_numpy(dists).to(query_input_ids.device)
            probs = F.softmax(dists, dim=-1)
        
        bs = query_features.shape[0]
        
        query_strs = self.tokenizer.batch_decode(query_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        all_lengths=[]
        new_input_ids = []

        all_attention_masks = []
        
        for i,query in enumerate(query_strs):
            query = query.split()
            origin_length=len(query)
            #all_origin_lengths.append(origin_length)  # query: str(protein_length), 没有+2
            
            inputs = []
            mask = []
            
            # first line of inputs is original query
            query_tokenized_result = self.tokenizer.encode_plus(
                    query_strs, padding='max_length', truncation=True, return_special_tokens_mask=True, max_length=max_length, is_split_into_words=True, return_tensors='np'
            )  
            inputs.append(query_tokenized_result['input_ids'])
            mask.append(query_tokenized_result['attention_mask'])
            all_lengths.append(origin_length)
            
            
            for index in re_indexes[i]:
                retrieve_seq = self.knn_dstore.get_retrieved_seqs(index)
                retrieve_seq  = list(re.sub(r"[UZOB]", "X", retrieve_seq ))
                all_lengths.append(len(retrieve_seq))
                
                tokenized_result = self.tokenizer.encode_plus(
                    retrieve_seq, padding='max_length', truncation=True, return_special_tokens_mask=True, max_length=max_length, is_split_into_words=True, return_tensors='np'
                )  # list (cls query sep context, pad to max length)
                inputs.append(tokenized_result['input_ids'])
                mask.append(tokenized_result['attention_mask'])
                
                
            inputs = np.concatenate(inputs)
            new_input_ids.append(inputs)
            mask = np.concatenate(mask)
            all_attention_masks.append(mask)
        
        new_input_ids = torch.tensor(np.array(new_input_ids), device=query_input_ids.device) #[batch_size, k]
        lengths = torch.tensor(all_lengths, device=query_input_ids.device) #[batch_size]  
        
        if return_mask:
            new_attention_masks = torch.tensor(np.array(all_attention_masks), device=query_input_ids.device)
            return new_input_ids, new_attention_masks, lengths, probs
        else:
            return new_input_ids, lengths, probs
        
    
    def retrieve_concat(self, query_features, query_input_ids, max_length=None, return_tensors="pt", return_mask=False, weight="uniform"):
        dists, re_indexes = self.knn_dstore.get_knns(query_features)
        if weight == "uniform":
            probs = torch.ones_like(dists) / dists.shape[1]
        elif weight == "distance":
            probs = self.knn_dstore.get_knn_prob(query_features, dists=dists, knns=re_indexes)
        
        bs = query_features.shape[0]
        
        query_strs = self.tokenizer.batch_decode(query_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        #print(''.join(query_strs[0].split()))
        all_origin_lengths=[]
        new_input_ids = []

        all_attention_masks = []

        for i,query in enumerate(query_strs):
            query = query.split()
            origin_length=len(query)
            all_origin_lengths.append(origin_length)  # query: str(protein_length), 没有+2
            
            #print(f"  origin_length = {origin_length}")
            
            concat_inputs = []
            concat_mask = []

            for index in re_indexes[i]:
                retrieve_seq = self.knn_dstore.get_retrieved_seqs(index)  
                #print(retrieve_seq)
                
                retrieve_seq  = list(re.sub(r"[UZOB]", "X", retrieve_seq ))
                
                #print(f"  retrieve_length = {len(retrieve_seq)}")
                
                tokenized_result = self.tokenizer.encode_plus(
                    query, retrieve_seq, padding='max_length', truncation='only_second', return_special_tokens_mask=True, max_length=max_length, is_split_into_words=True, return_tensors='np'
                )  # list (cls query sep context, pad to max length), double check evaluation code
                
                #print(f"  tokenized_length = {len(tokenized_result['input_ids'][0])}")
                
                concat_inputs.append(tokenized_result['input_ids'])

                concat_mask.append(tokenized_result['attention_mask'])

            concat_inputs = np.concatenate(concat_inputs)
            new_input_ids.append(concat_inputs) 
            concat_mask = np.concatenate(concat_mask)
            all_attention_masks.append(concat_mask)

        new_input_ids = torch.tensor(np.array(new_input_ids), device=query_input_ids.device) #[batch_size, k]
        origin_lengths = torch.tensor(all_origin_lengths, device=query_input_ids.device) #[batch_size]
        
        if return_mask:
            new_attention_masks = torch.tensor(np.array(all_attention_masks), device=query_input_ids.device)
            return new_input_ids, new_attention_masks, origin_lengths, probs
        else:
            return new_input_ids, origin_lengths, probs


