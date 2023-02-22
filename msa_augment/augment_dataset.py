from benchmark.base_dataset import LMDBDataset, DataProcessor
from pathlib import Path
# from plistlib import Dict
from typing import Union

import pickle as pkl
import lmdb
import numpy as np
import pandas as pd
import re
import torch
from scipy.spatial.distance import squareform, pdist
from tape.datasets import pad_sequences, dataset_factory
from torch.utils.data import Dataset
import os
import random
from benchmark.utils import get_features
from scipy.spatial.distance import cdist
from tqdm import tqdm
def greedy_select(msa, num_seqs: int, mode: str = "max"):
    assert mode in ("max", "min")
    if len(msa) <= num_seqs:
        return msa

    array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)

    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))
    for _ in range(num_seqs - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    indices = sorted(indices)
    return [msa[idx] for idx in indices]

def random_select(msa,num_seqs):
    if len(msa) <= num_seqs:
        return msa
    else:
        result=[]
        result.append(msa[0])
        others=torch.utils.data.random_split(msa[1:], [num_seqs-1,len(msa)-num_seqs])[0]
        for data in others:
            result.append(data)
        return result
    
def first_select(msa,num_seqs):
    if len(msa) <= num_seqs:
        return msa
    else:
        return msa[0:num_seqs]

class AugmentFluorescenceProgress(DataProcessor):
    def __init__(self, max_len, concat_max_len, tokenizer, need_features=False, preprocess_device='0',select_method='random',num_msa=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device
        self.select_method=select_method
        self.num_msa=num_msa
        self.concat_max_len = concat_max_len

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = AugmentFluorescenceDataset(data_dir, split='train', max_len=self.max_len, concat_max_len=self.concat_max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = AugmentFluorescenceDataset(data_dir, split='valid', max_len=self.max_len, concat_max_len=self.concat_max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        if data_cat is not None:
            dataset = AugmentFluorescenceDataset(data_dir, split=data_cat, max_len=self.max_len,concat_max_len=self.concat_max_len,  tokenizer=self.tokenizer,in_memory=in_memory,
                                          need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        else:
            dataset = AugmentFluorescenceDataset(data_dir, split='test', max_len=self.max_len, concat_max_len=self.concat_max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                          need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_labels(self):
        return list(range(1))


class AugmentSecondaryStructureProcessor3(DataProcessor):
    def __init__(self, max_len, concat_max_len, tokenizer, need_features=False, preprocess_device='0',select_method='random',num_msa=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device
        self.select_method=select_method
        self.num_msa=num_msa
        self.concat_max_len = concat_max_len

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = AugmentSecondaryStructureDataset3(data_dir, split='train', max_len=self.max_len, concat_max_len=self.concat_max_len, tokenizer=self.tokenizer,
                                             target='ss3', in_memory=in_memory,
                                             need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = AugmentSecondaryStructureDataset3(data_dir, split='valid', max_len=self.max_len, concat_max_len=self.concat_max_len, tokenizer=self.tokenizer,
                                             target='ss3', in_memory=in_memory,
                                             need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        dataset = AugmentSecondaryStructureDataset3(data_dir, split=data_cat, max_len=self.max_len, concat_max_len=self.concat_max_len,  tokenizer=self.tokenizer,
                                             target='ss3', in_memory=in_memory,
                                             need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_labels(self):
        return list(range(3))


class AugmentSecondaryStructureProcessor8(DataProcessor):
    def __init__(self, max_len, concat_max_len, tokenizer, need_features=False, preprocess_device='0',select_method='random',num_msa=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device
        self.select_method=select_method
        self.num_msa=num_msa
        self.concat_max_len = concat_max_len

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = AugmentSecondaryStructureDataset8(data_dir, split='train', max_len=self.max_len, concat_max_len=self.concat_max_len, tokenizer=self.tokenizer,
                                             target='ss8', in_memory=in_memory,
                                             need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = AugmentSecondaryStructureDataset8(data_dir, split='valid', max_len=self.max_len, concat_max_len=self.concat_max_len, tokenizer=self.tokenizer,
                                             target='ss8', in_memory=in_memory,
                                             need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        dataset = AugmentSecondaryStructureDataset8(data_dir, split=data_cat, max_len=self.max_len, concat_max_len=self.concat_max_len, tokenizer=self.tokenizer,
                                             target='ss8', in_memory=in_memory,
                                             need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_labels(self):
        return list(range(8))


class AugmentContactProgress(DataProcessor):
    def __init__(self, max_len, concat_max_len, tokenizer, need_features=False, preprocess_device='0',select_method='random',num_msa=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.concat_max_len = concat_max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = AugmentProteinnetDataset(data_dir, split='train', max_len=self.max_len, concat_max_len=self.concat_max_len, tokenizer=self.tokenizer,
                                    need_features=self.need_features, preprocess_device=self.prepocess_device,select_method='random',num_msa=128)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = AugmentProteinnetDataset(data_dir, split='valid', max_len=self.max_len, concat_max_len=self.concat_max_len, tokenizer=self.tokenizer,
                                    need_features=self.need_features, preprocess_device=self.prepocess_device,select_method='random',num_msa=128)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        if data_cat is not None:
            dataset = AugmentProteinnetDataset(data_dir, split=data_cat, max_len=self.max_len, concat_max_len=self.concat_max_len, tokenizer=self.tokenizer,
                                        need_features=self.need_features, preprocess_device=self.prepocess_device,select_method='random',num_msa=128)
        else:
            dataset = AugmentProteinnetDataset(data_dir, split='test', max_len=self.max_len, concat_max_len=self.concat_max_len, tokenizer=self.tokenizer,
                                        need_features=self.need_features, preprocess_device=self.prepocess_device,select_method='random',num_msa=128)
        return dataset

    def get_labels(self):
        return list(range(2))


class AugmentStabilityProgress(DataProcessor):
    def __init__(self, max_len, concat_max_len, tokenizer, need_features=False, preprocess_device='0',select_method='random',num_msa=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.concat_max_len = concat_max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device
        self.select_method=select_method
        self.num_msa=num_msa

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = AugmentStabilityDataset(data_dir, split='train', max_len=self.max_len, concat_max_len = self.concat_max_len, tokenizer=self.tokenizer, in_memory=in_memory,
                                   need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = AugmentStabilityDataset(data_dir, split='valid', max_len=self.max_len, concat_max_len = self.concat_max_len, tokenizer=self.tokenizer, in_memory=in_memory,
                                   need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        if data_cat is not None:
            dataset = AugmentStabilityDataset(data_dir, split=data_cat, max_len=self.max_len, concat_max_len = self.concat_max_len, tokenizer=self.tokenizer, in_memory=in_memory,
                                       need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        else:
            dataset = AugmentStabilityDataset(data_dir, split='test', max_len=self.max_len,concat_max_len = self.concat_max_len,  tokenizer=self.tokenizer, in_memory=in_memory,
                                       need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_labels(self):
        return list(range(1))


class AugmentRemoteHomologyProgress(DataProcessor):
    def __init__(self, max_len, concat_max_len, tokenizer, need_features=False, preprocess_device='0',select_method='random',num_msa=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.concat_max_len = concat_max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device
        self.select_method=select_method
        self.num_msa=num_msa

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = AugmentRemoteHomologyDataset(data_dir, split='train', max_len=self.max_len, concat_max_len = self.concat_max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                        need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = AugmentRemoteHomologyDataset(data_dir, split='valid', max_len=self.max_len, concat_max_len = self.concat_max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                        need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        if data_cat is not None:
            dataset = AugmentRemoteHomologyDataset(data_dir, split=data_cat, max_len=self.max_len, concat_max_len = self.concat_max_len,  tokenizer=self.tokenizer,in_memory=in_memory,
                                            need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        else:
            dataset = AugmentRemoteHomologyDataset(data_dir, split='test', max_len=self.max_len, concat_max_len = self.concat_max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                            need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_labels(self):
        return list(range(1195))


class AugmentProteinnetDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 max_len: int,
                 concat_max_len: int, 
                 tokenizer,
                 need_features=False,
                 preprocess_device='0',
                 select_method='random',
                 num_msa=10
                 ):

        if split not in ('train', 'train_unfiltered', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'train_unfiltered', 'valid', 'test']")

        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'proteinnet/proteinnet_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file)
        self.max_len = max_len
        self.concat_max_len = concat_max_len
        self.need_features = need_features

        if self.need_features:
            seqs = [item['primary'] for item in self.data]
            store_path = str(data_path / f'proteinnet/proteinnet_{split}.fvecs')
            self.features = get_features(seqs, store_path, preprocess_device)

        self.select_fn = {'random': random_select, 'greedy': greedy_select}[select_method]
        self.num_msa = num_msa

    def __len__(self) -> int:
        return len(self.data)

    '''
    def __getitem__(self, index: int):
        max_len = self.max_len
        item = self.data[index]
        seq = list(re.sub(r"[UZOB]", "X", item['primary']))
        protein_length = min(len(seq), max_len-2)
        seq = seq[:protein_length]
        token_ids = self.tokenizer(seq, is_split_into_words=True)
        token_ids = np.asarray(token_ids['input_ids'], dtype=int)
        input_mask = np.ones_like(token_ids)

        valid_mask = np.array(item['valid_mask'])[:protein_length]
        contact_map = np.less(squareform(pdist(item['tertiary'])), 8.0).astype(np.int64)
        contact_map = contact_map[:protein_length, :protein_length]


        yind, xind = np.indices(contact_map.shape)
        invalid_mask = ~(valid_mask[:, None] & valid_mask[None, :])
        invalid_mask |= np.abs(yind - xind) < 6
        contact_map[invalid_mask] = -1


        return token_ids, input_mask, contact_map, protein_length
    '''

    def __getitem__(self, index: int):
        try:
            item = self.data[index]
        except:
            return self.__getitem__(np.random.randint(len(self.data)))

        primary_sequence = item['primary']
        primary_sequence = re.sub(r"[a-zUZOB]", "X", primary_sequence)
        
        protein_length = min(len(primary_sequence), self.max_len-2)
        primary_sequence = primary_sequence[:protein_length]
        
        msa_lists = item['msa'][1:]
        
        token_ids = []
        input_mask  = []
        
        if len(msa_lists) > self.num_msa:
            msa_lists = random.sample(msa_lists, self.num_msa)
        
        for data in msa_lists:
            data=re.sub(r"[a-zUZOB]", "X", data)
            data=re.sub(r"-", "", data)
            
            tokenized_result = self.tokenizer.encode_plus(
                list(primary_sequence), list(data), padding='max_length', truncation=True, return_special_tokens_mask=True, max_length=self.max_len, is_split_into_words=True, return_tensors='np'
            ) 
            
            token_ids.append(tokenized_result['input_ids'])
            input_mask.append(tokenized_result['attention_mask'])
            
        token_ids = np.array(token_ids, dtype=np.int64)
        input_mask = np.array(input_mask, dtype=np.int64)


        valid_mask = item['valid_mask']
        valid_mask = np.array(valid_mask)[:protein_length]
        # print("type:", type(valid_mask))
        # print("valid_mask", valid_mask)
        contact_map = np.less(squareform(pdist(torch.tensor(item['tertiary']))), 8.0).astype(np.int64)
        contact_map = contact_map[:protein_length, :protein_length]

        yind, xind = np.indices(contact_map.shape)
        # DEL
        invalid_mask = ~(valid_mask[:, None] & valid_mask[None, :])
        invalid_mask |= np.abs(yind - xind) < 6
        contact_map[invalid_mask] = -1

        if not self.need_features:
            return token_ids, protein_length, input_mask, contact_map, 
        else:
            return token_ids, protein_length, input_mask, contact_map, self.features[index]

    def collate_fn(self, batch):
        if not self.need_features:
            input_ids, protein_length, input_mask, contact_labels = tuple(zip(*batch))
        else:
            input_ids, protein_length, input_mask, contact_labels, features = tuple(zip(*batch))

        input_ids = torch.from_numpy(input_ids)
        input_mask = torch.from_numpy(input_ids)
        contact_labels = torch.from_numpy(pad_sequences(contact_labels, -1))
        protein_length = torch.LongTensor(protein_length)  # type: ignore

        result = {'input_ids': input_ids,
                  'attention_mask': input_mask,
                  'labels': contact_labels,
                  'origin_length': protein_length}

        if self.need_features:
            features = torch.from_numpy(np.array(features))
            result['features'] = features

        return result


class AugmentFluorescenceDataset(Dataset):
    def __init__(self, data_path, split, max_len, concat_max_len, tokenizer,in_memory,
                 need_features=False,
                 preprocess_device='0',
                 select_method='random',
                 num_msa=10):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.concat_max_len = concat_max_len

        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test'")

        data_file = f'fluorescence/fluorescence_{split}.lmdb'
        self.data = LMDBDataset(data_file=os.path.join(data_path, data_file), in_memory=in_memory)
        # self.seqs, self.labels = self.get_data(data_file)

        self.need_features = need_features
        if self.need_features:
            seqs = [item['primary'] for item in self.data]
            store_path = f'{data_path}/fluorescence/fluorescence_{split}.fvecs'
            self.features = get_features(seqs, store_path, preprocess_device)


        self.select_fn = {'random': random_select, 'greedy': greedy_select}[select_method]
        self.num_msa = num_msa


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.data[index]
        except:
            return self.__getitem__(np.random.randint(len(self.data)))
        
        primary_sequence = item['primary']
        primary_sequence = re.sub(r"[a-zUZOB]", "X", primary_sequence)
        
        protein_length = min(len(primary_sequence), self.max_len-2)
        primary_sequence = primary_sequence[:protein_length]
        
        msa_lists = item['msa'][0:] # if there is no msa [input [sep] input]
        
        token_ids = []
        input_mask  = []
        
        if len(msa_lists) > self.num_msa:
            msa_lists = random.sample(msa_lists, self.num_msa)
        
        for data in msa_lists:
            data=re.sub(r"[a-zUZOB]", "X", data)
            data=re.sub(r"-", "", data)
            
            tokenized_result = self.tokenizer.encode_plus(
                list(primary_sequence), list(data), padding='max_length', truncation=True, return_special_tokens_mask=True, max_length=self.concat_max_len, is_split_into_words=True, return_tensors='np'
            ) 
            
            token_ids.append(tokenized_result['input_ids'].squeeze())
            input_mask.append(tokenized_result['attention_mask'].squeeze())
            
        token_ids = np.array(token_ids, dtype=np.int64)
        input_mask = np.array(input_mask, dtype=np.int64)
        
        
        # pad with -1s because of cls/sep tokens
        labels = np.asarray(item['log_fluorescence'], np.float32)


        if not self.need_features:
            return token_ids, protein_length, input_mask, labels
        else:
            return token_ids, protein_length, input_mask, labels, self.features[index]



    def collate_fn(self, batch):
        if not self.need_features:
            input_ids, protein_length, input_mask, ss_label = tuple(zip(*batch))
        else:
            input_ids, protein_length, input_mask, ss_label, features = tuple(zip(*batch))
        input_ids = torch.cat([torch.from_numpy(input).unsqueeze(0) for input in input_ids], 0)
        attention_mask = torch.cat([torch.from_numpy(mask).unsqueeze(0) for mask in input_mask], 0)
        labels = torch.from_numpy(np.concatenate(ss_label,0))
        protein_length = torch.LongTensor(protein_length)  # type: ignore

        output = {'input_ids': input_ids,
                  'attention_mask': attention_mask,
                  'labels': labels,
                  'origin_length': protein_length}

        return output


class AugmentStabilityDataset(Dataset):
    def __init__(self, data_path, split, max_len, concat_max_len, tokenizer,in_memory,
                 need_features=False,
                 preprocess_device='0',
                 select_method='random',
                 num_msa=10):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.concat_max_len = concat_max_len

        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test'")

        data_file = f'stability/stability_{split}.lmdb'
        self.data = LMDBDataset(data_file=os.path.join(data_path, data_file), in_memory=in_memory)
        # self.seqs, self.labels = self.get_data(data_file)

        self.need_features = need_features
        if self.need_features:
            seqs = [item['primary'] for item in self.data]
            store_path = f'{data_path}/stability/stability_{split}.fvecs'
            self.features = get_features(seqs, store_path, preprocess_device)


        self.select_fn = {'random': random_select, 'greedy': greedy_select,'first':first_select}[select_method]
        self.num_msa = num_msa



    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.data[index]
        except:
            return self.__getitem__(np.random.randint(len(self.data)))

        primary_sequence = item['primary']
        primary_sequence = re.sub(r"[a-zUZOB]", "X", primary_sequence)
        
        protein_length = min(len(primary_sequence), self.max_len-2)
        primary_sequence = primary_sequence[:protein_length]
        
        msa_lists = item['msa'] # if there is no msa [input [sep] input]
        
        token_ids = []
        input_mask  = []
        
        if len(msa_lists) > self.num_msa:
            msa_lists = msa_lists[:self.num_msa]
        
        for data in msa_lists:
            data=re.sub(r"[a-zUZOB]", "X", data)
            data=re.sub(r"-", "", data)
            
            tokenized_result = self.tokenizer.encode_plus(
                list(primary_sequence), list(data), padding='max_length', truncation='only_second', return_special_tokens_mask=True, max_length=self.concat_max_len, is_split_into_words=True, return_tensors='np'
            ) 
            
            token_ids.append(tokenized_result['input_ids'].squeeze())
            input_mask.append(tokenized_result['attention_mask'].squeeze())
            
        token_ids = np.array(token_ids, dtype=np.int64)
        input_mask = np.array(input_mask, dtype=np.int64)
        

        # pad with -1s because of cls/sep tokens
        labels = np.asarray(item['stability_score'], np.float32)


        if not self.need_features:
            return token_ids, protein_length, input_mask, labels
        else:
            return token_ids, protein_length, input_mask, labels, self.features[index]



    def collate_fn(self, batch):
        if not self.need_features:
            input_ids, protein_length, input_mask, ss_label = tuple(zip(*batch))
        else:
            input_ids, protein_length, input_mask, ss_label, features = tuple(zip(*batch))
        input_ids = torch.cat([torch.from_numpy(input).unsqueeze(0) for input in input_ids], 0)
        attention_mask = torch.cat([torch.from_numpy(mask).unsqueeze(0) for mask in input_mask], 0)
        labels = torch.from_numpy(np.concatenate(ss_label,0))
        protein_length = torch.LongTensor(protein_length)  # type: ignore

        output = {'input_ids': input_ids,
                  'attention_mask': attention_mask,
                  'labels': labels,
                  'origin_length': protein_length}

        return output



class AugmentRemoteHomologyDataset(Dataset):
    def __init__(self, data_path, split, max_len, concat_max_len, tokenizer,in_memory,
                 need_features=False,
                 preprocess_device='0',
                 select_method='random',
                 num_msa=10):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.concat_max_len = concat_max_len

        if split not in ('train', 'valid', 'test_fold_holdout',
                         'test_family_holdout', 'test_superfamily_holdout'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test_fold_holdout', "
                             f"'test_family_holdout', 'test_superfamily_holdout']")

        data_file = f'remote_homology/remote_homology_{split}.lmdb'
        self.data = LMDBDataset(data_file=os.path.join(data_path, data_file), in_memory=in_memory)
        # self.seqs, self.labels = self.get_data(data_file)

        self.need_features = need_features
        if self.need_features:
            seqs = [item['primary'] for item in self.data]
            store_path = f'{data_path}/remote_homology/remote_homology{split}.fvecs'
            self.features = get_features(seqs, store_path, preprocess_device)


        self.select_fn = {'random': random_select, 'greedy': greedy_select,'first':first_select}[select_method]
        self.num_msa = num_msa

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.data[index]
        except:
            return self.__getitem__(np.random.randint(len(self.data)))

        primary_sequence = item['primary']
        primary_sequence = re.sub(r"[a-zUZOB]", "X", primary_sequence)
        
        protein_length = min(len(primary_sequence), self.max_len-2)
        primary_sequence = primary_sequence[:protein_length]
        
        msa_lists = item['msa'] # if there is no msa [input [sep] input]
        
        token_ids = []
        input_mask  = []
        
        if len(msa_lists) > self.num_msa:
            msa_lists = msa_lists[:self.num_msa]
        
        for data in msa_lists:
            data=re.sub(r"[a-zUZOB]", "X", data)
            data=re.sub(r"-", "", data)
            
            tokenized_result = self.tokenizer.encode_plus(
                list(primary_sequence), list(data), padding='max_length', truncation='only_second', return_special_tokens_mask=True, max_length=self.concat_max_len, is_split_into_words=True, return_tensors='np'
            ) 
            
            token_ids.append(tokenized_result['input_ids'].squeeze())
            input_mask.append(tokenized_result['attention_mask'].squeeze())
            
        token_ids = np.array(token_ids, dtype=np.int64)
        input_mask = np.array(input_mask, dtype=np.int64)
        
        # pad with -1s because of cls/sep tokens
        labels = np.asarray([item['fold_label']], np.int64)


        if not self.need_features:
            return token_ids, protein_length, input_mask, labels
        else:
            return token_ids, protein_length, input_mask, labels, self.features[index]



    def collate_fn(self, batch):
        if not self.need_features:
            input_ids, protein_length, input_mask, ss_label = tuple(zip(*batch))
        else:
            input_ids, protein_length, input_mask, ss_label, features = tuple(zip(*batch))
        input_ids = torch.cat([torch.from_numpy(input).unsqueeze(0) for input in input_ids], 0)
        attention_mask = torch.cat([torch.from_numpy(mask).unsqueeze(0) for mask in input_mask], 0)
        labels = torch.from_numpy(np.concatenate(ss_label,0))
        protein_length = torch.LongTensor(protein_length)  # type: ignore
        
        output = {'input_ids': input_ids,
                  'attention_mask': attention_mask,
                  'labels': labels,
                  'origin_length':protein_length}

        return output





class AugmentSecondaryStructureDataset3(Dataset):
    def __init__(
            self,
            data_path,
            split,
            max_len,
            concat_max_len,
            tokenizer,
            in_memory,
            target='ss3',
            need_features=False,
            preprocess_device='0',
            select_method='random',
            num_msa=10
    ):
        self.tokenizer = tokenizer
        data_file = f'secondary_structure/secondary_structure_{split}.lmdb'
        self.data = LMDBDataset(data_file=os.path.join(data_path, data_file), in_memory=in_memory)

        self.select_fn={'random':random_select,'greedy':greedy_select}[select_method]
        self.target = target
        self.max_len = max_len
        self.concat_max_len = concat_max_len
        
        self.ignore_index: int = -100
        self.num_msa=num_msa

        self.need_features = need_features

        if self.need_features:
            seqs = [item['primary'] for item in self.data]
            store_path = data_path + '/' + f'secondary_structure/secondary_structure_{split}.fvecs'
            self.features = get_features(seqs, store_path, preprocess_device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        try:
            item = self.data[index]
        except:
            return self.__getitem__(np.random.randint(len(self.data)))
        primary_sequence = item['primary']
        primary_sequence = re.sub(r"[a-zUZOB]", "X", primary_sequence)
        
        protein_length = min(len(primary_sequence), self.max_len-2)
        primary_sequence = primary_sequence[:protein_length]
        
        msa_lists = item['msa'][1:]
        
        token_ids = []
        input_mask  = []
        
        msa_lists = random.sample(msa_lists, self.num_msa)
        
        for data in msa_lists:
            data=re.sub(r"[a-zUZOB]", "X", data)
            data=re.sub(r"-", "", data)
            
            tokenized_result = self.tokenizer.encode_plus(
                list(primary_sequence), list(data), padding='max_length', truncation=True, return_special_tokens_mask=True, max_length=self.max_len, is_split_into_words=True, return_tensors='np'
            ) 
            
            token_ids.append(tokenized_result['input_ids'])
            input_mask.append(tokenized_result['attention_mask'])
            
        token_ids = np.array(token_ids, dtype=np.int64)
        input_mask = np.array(input_mask, dtype=np.int64)
        

        if len(item['primary']) > self.max_len:
            item['primary'] = item['primary'][:self.max_len]  # str(protein_length)
            item['ss3'] = item['ss3'][:self.max_len]  # str(protein_length)

        # pad with -1s because of cls/sep tokens
        labels = np.asarray(item['ss3'], np.int64)
        labels = np.pad(labels, (1, token_ids.shape[-1]-len(labels)-1), 'constant', constant_values=self.ignore_index)  # list(protein_length+2)
        labels=np.reshape(labels,[1,-1])

        if not self.need_features:
            return token_ids, protein_length, input_mask, labels
        else:
            return token_ids, protein_length, input_mask, labels, self.features[index]

    def collate_fn(self, batch):
        if not self.need_features:
            input_ids, protein_length, input_mask, ss_label = tuple(zip(*batch))
        else:
            input_ids, protein_length, input_mask, ss_label, features = tuple(zip(*batch))
        input_ids = torch.from_numpy(input_ids)
        attention_mask = None
        labels = torch.from_numpy(np.concatenate(ss_label,0))
        protein_length = torch.LongTensor(protein_length)  # type: ignore
        
        output = {'input_ids': input_ids,
                  'attention_mask': attention_mask,
                  'labels': labels,
                  'origin_length':protein_length}
        if self.need_features:
            features = torch.from_numpy(np.array(features))
            output['features'] = features
        return output


class AugmentSecondaryStructureDataset8(Dataset):
    def __init__(
            self,
            data_path,
            split,
            max_len,
            concat_max_len,
            tokenizer,
            in_memory,
            target='ss8',
            need_features=False,
            preprocess_device='0',
            select_method='random',
            num_msa=10

    ):
        self.tokenizer = tokenizer
        data_file = f'secondary_structure/secondary_structure_{split}.lmdb'
        self.data = LMDBDataset(data_file=os.path.join(data_path, data_file), in_memory=in_memory)
        self.target = target
        self.max_len = max_len
        self.concat_max_len = concat_max_len
        self.ignore_index: int = -100

        self.need_features = need_features

        if self.need_features:
            seqs = [item['primary'] for item in self.data]
            store_path = data_path + '/' + f'secondary_structure/secondary_structure_{split}.fvecs'
            self.features = get_features(seqs, store_path, preprocess_device)

        self.select_fn = {'random': random_select, 'greedy': greedy_select}[select_method]
        self.num_msa = num_msa

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        try:
            item = self.data[index]
        except:
            return self.__getitem__(np.random.randint(len(self.data)))

        primary_sequence = item['primary']
        primary_sequence = re.sub(r"[a-zUZOB]", "X", primary_sequence)
        
        protein_length = min(len(primary_sequence), self.max_len-2)
        primary_sequence = primary_sequence[:protein_length]
        
        msa_lists = item['msa'][1:]
        
        token_ids = []
        input_mask  = []
        
        msa_lists = random.sample(msa_lists, self.num_msa)
        
        for data in msa_lists:
            data=re.sub(r"[a-zUZOB]", "X", data)
            data=re.sub(r"-", "", data)
            
            tokenized_result = self.tokenizer.encode_plus(
                list(primary_sequence), list(data), padding='max_length', truncation=True, return_special_tokens_mask=True, max_length=self.max_len, is_split_into_words=True, return_tensors='np'
            ) 
            
            token_ids.append(tokenized_result['input_ids'])
            input_mask.append(tokenized_result['attention_mask'])
            
        token_ids = np.array(token_ids, dtype=np.int64)
        input_mask = np.array(input_mask, dtype=np.int64)
        #label
        if len(item['primary']) > self.max_len:
            item['primary'] = item['primary'][:self.max_len]  # str(protein_length)
            item['ss8'] = item['ss8'][:self.max_len]  # str(protein_length)

        # pad with -1s because of cls/sep tokens
        labels = np.asarray(item['ss8'], np.int64)
        labels = np.pad(labels, (1, token_ids.shape[-1]-len(labels)-1), 'constant', constant_values=self.ignore_index)  # list(protein_length+2)
        labels=np.reshape(labels,[1,-1])

        if not self.need_features:
            return token_ids, protein_length, input_mask, labels
        else:
            return token_ids, protein_length, input_mask, labels, self.features[index]

    def collate_fn(self, batch):
        if not self.need_features:
            input_ids, protein_length, input_mask, ss_label = tuple(zip(*batch))
        else:
            input_ids, protein_length, input_mask, ss_label, features = tuple(zip(*batch))
        input_ids = torch.from_numpy(np.concatenate(input_ids,0))
        attention_mask = None
        labels = torch.from_numpy(np.concatenate(ss_label,0))
        protein_length = torch.LongTensor(protein_length)  # type: ignore
        
        output = {'input_ids': input_ids,
                  'attention_mask': attention_mask,
                  'labels': labels,
                  'origin_length': protein_length}
        if self.need_features:
            features = torch.from_numpy(np.array(features))
            output['features'] = features
        return output






