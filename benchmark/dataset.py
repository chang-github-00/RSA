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
from .utils import get_features
from .base_dataset import LMDBDataset, DataProcessor
from msa import MSAFluorescenceProgress, MSASecondaryStructureProcessor3, MSASecondaryStructureProcessor8, MSAContactProgress, MSAStabilityProgress, MSARemoteHomologyProgress,MSAProteinProteinInteractionProgress,MSASubcellularLocalizationProgress
from msa_augment import AugmentContactProgress, AugmentFluorescenceProgress, AugmentRemoteHomologyProgress, AugmentSecondaryStructureProcessor3, AugmentSecondaryStructureProcessor8, AugmentStabilityProgress
# def pad_sequences(sequences, constant_value=0, dtype=None):
#     batch_size = len(sequences)
#     shape = [batch_size] + np.max([seq.shape for seq in sequences], axis=0).tolist()
#
#     if dtype == None:
#         dtype = sequences[0].dtype
#
#     array = np.full(shape, constant_value, dtype)
#
#     for arr, seq in zip(array, sequences):
#         arrslice = tuple(slice(dim) for dim in seq.shape)
#         arr[arrslice] = seq
#
#     return array


class FluorescenceProgress(DataProcessor):
    def __init__(self, max_len, tokenizer, need_features=False, preprocess_device='0',feature_type='esm', **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device
        self.feature_type = feature_type

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = FluorescenceDataset(data_dir, split='train', max_len = self.max_len , tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device, 
                                      feature_type=self.feature_type)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = FluorescenceDataset(data_dir, split='valid', max_len = self.max_len ,tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device,
                                      feature_type=self.feature_type)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        if data_cat is not None:
            dataset = FluorescenceDataset(data_dir, split=data_cat, max_len = self.max_len ,tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device,
                                      feature_type=self.feature_type)
        else:
            dataset = FluorescenceDataset(data_dir, split='test', max_len = self.max_len ,tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device,
                                      feature_type=self.feature_type)
        return dataset

    def get_labels(self):
        return list(range(1))


class SecondaryStructureProcessor3(DataProcessor):
    def __init__(self, max_len, tokenizer, need_features=False, preprocess_device='0', feature_type='esm', **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device
        self.feature_type = feature_type

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = SecondaryStructureDataset3(data_dir, split='train', max_len = self.max_len, tokenizer=self.tokenizer, target='ss3', in_memory=in_memory,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device,
                                      feature_type=self.feature_type)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = SecondaryStructureDataset3(data_dir, split='valid',  max_len = self.max_len, tokenizer=self.tokenizer, target='ss3', in_memory=in_memory,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device,
                                      feature_type=self.feature_type)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        dataset = SecondaryStructureDataset3(data_dir, split=data_cat,  max_len = self.max_len, tokenizer=self.tokenizer, target='ss3', in_memory=in_memory,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device,
                                      feature_type=self.feature_type)
        return dataset

    def get_labels(self):
        return list(range(3))


class SecondaryStructureProcessor8(DataProcessor):
    def __init__(self, max_len, tokenizer, need_features=False, preprocess_device='0', feature_type='esm', **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device
        self.feature_type = feature_type

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = SecondaryStructureDataset8(data_dir, split='train', max_len = self.max_len, tokenizer=self.tokenizer, target='ss8', in_memory=in_memory,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device, feature_type = self.feature_type)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = SecondaryStructureDataset8(data_dir, split='valid',  max_len = self.max_len,tokenizer=self.tokenizer, target='ss8', in_memory=in_memory,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device, feature_type = self.feature_type)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        dataset = SecondaryStructureDataset8(data_dir, split=data_cat,  max_len = self.max_len,tokenizer=self.tokenizer, target='ss8', in_memory=in_memory,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device, feature_type = self.feature_type)
        return dataset

    def get_labels(self):
        return list(range(8))


class ContactProgress(DataProcessor):
    def __init__(self, max_len, tokenizer, need_features=False, preprocess_device='0', feature_type='esm', **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device
        self.feature_type = feature_type
        
    def get_train_examples(self, data_dir, in_memory=True):
        dataset = ProteinnetDataset(data_dir, split='train', max_len = self.max_len , tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device,
                                      feature_type=self.feature_type)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = ProteinnetDataset(data_dir, split='valid', max_len = self.max_len, tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device,
                                      feature_type=self.feature_type)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        if data_cat is not None:
            dataset = ProteinnetDataset(data_dir, split=data_cat, max_len = self.max_len, tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device,
                                      feature_type=self.feature_type)
        else:
            dataset = ProteinnetDataset(data_dir, split='test', max_len = self.max_len,tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device,
                                      feature_type=self.feature_type)
        return dataset

    def get_labels(self):
        return list(range(2))


class StabilityProgress(DataProcessor):
    def __init__(self, max_len, tokenizer, need_features=False, preprocess_device='0', feature_type='esm', **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device
        self.feature_type = feature_type

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = StabilityDataset(data_dir, split='train', max_len = self.max_len, tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device,
                                      feature_type = self.feature_type)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = StabilityDataset(data_dir, split='valid', max_len = self.max_len, tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device,
                                      feature_type = self.feature_type)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        if data_cat is not None:
            dataset = StabilityDataset(data_dir, split=data_cat, max_len = self.max_len, tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device,
                                      feature_type = self.feature_type)
        else:
            dataset = StabilityDataset(data_dir, split='test', max_len = self.max_len, tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device,
                                      feature_type = self.feature_type)
        return dataset

    def get_labels(self):
        return list(range(1))


class RemoteHomologyProgress(DataProcessor):
    def __init__(self, max_len, tokenizer, need_features=False, preprocess_device='0', feature_type='esm', **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device
        self.feature_type = feature_type

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = RemoteHomologyDataset(data_dir, split='train', max_len = self.max_len, tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device, 
                                      feature_type = self.feature_type)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = RemoteHomologyDataset(data_dir, split='valid', max_len = self.max_len, tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device, 
                                      feature_type = self.feature_type)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        if data_cat is not None:
            dataset = RemoteHomologyDataset(data_dir, split=data_cat, max_len = self.max_len, tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device, 
                                      feature_type = self.feature_type)
        else:
            dataset = RemoteHomologyDataset(data_dir, split='test', max_len = self.max_len, tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device, 
                                      feature_type = self.feature_type)
        return dataset

    def get_labels(self):
        return list(range(1195))

class SubcellularLocalizationProgress(DataProcessor):
    def __init__(self, max_len, tokenizer, need_features=False, preprocess_device='0', feature_type='esm', **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device
        self.feature_type = feature_type

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = SubcellularLocalizationDataset(data_dir, split='train', max_len = self.max_len, tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device, 
                                      feature_type = self.feature_type)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = SubcellularLocalizationDataset(data_dir, split='valid', max_len = self.max_len, tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device, 
                                      feature_type = self.feature_type)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        if data_cat is not None:
            dataset = SubcellularLocalizationDataset(data_dir, split=data_cat, max_len = self.max_len, tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device, 
                                      feature_type = self.feature_type)
        else:
            dataset = SubcellularLocalizationDataset(data_dir, split='test', max_len = self.max_len, tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device, 
                                      feature_type = self.feature_type)
        return dataset

    def get_labels(self):
        return list(range(10))


class ProteinProteinInteractionProgress(DataProcessor):
    def __init__(self, max_len, tokenizer, need_features=False, preprocess_device='0', feature_type='esm', **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device
        self.feature_type = feature_type

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = ProteinProteinInteractionDataset(data_dir, split='train', max_len = self.max_len, tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device, 
                                      feature_type = self.feature_type)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = ProteinProteinInteractionDataset(data_dir, split='valid', max_len = self.max_len, tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device, 
                                      feature_type = self.feature_type)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        if data_cat is not None:
            dataset = ProteinProteinInteractionDataset(data_dir, split=data_cat, max_len = self.max_len, tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device, 
                                      feature_type = self.feature_type)
        else:
            dataset = ProteinProteinInteractionDataset(data_dir, split='test', max_len = self.max_len, tokenizer=self.tokenizer,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device, 
                                      feature_type = self.feature_type)
        return dataset

    def get_labels(self):
        return list(range(2))




class ProteinnetDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 max_len: int,
                 tokenizer,
                 need_features=False,
                 preprocess_device='0',
                 feature_type='esm'
                 ):

        if split not in ('train', 'train_unfiltered', 'valid', 'test', 'de_novo'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'train_unfiltered', 'valid', 'test', 'de_novo]")

        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'proteinnet/proteinnet_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file)
        self.max_len = max_len
        self.need_features = need_features
        self.feature_type = feature_type
        
        if self.need_features:
            seqs = [item['primary'] for item in self.data]
            store_path = str(data_path/f'proteinnet/proteinnet_{split}.fvecs')
            self.features = get_features(seqs, store_path, preprocess_device, feature_type=self.feature_type)
        
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
        item = self.data[index]
        max_len = self.max_len
        
        seq = list(re.sub(r"[UZOB]", "X", item['primary']))
        protein_length = min(len(seq), max_len-2)
        
        seq = seq[:protein_length]
        
        token_ids = self.tokenizer(seq, is_split_into_words=True)
        token_ids = np.asarray(token_ids['input_ids'], dtype=int)
        
        #if protein_length > 1000:
        #    print(seq)
        input_mask = np.ones_like(token_ids)

        valid_mask = item['valid_mask']
        valid_mask = np.array(valid_mask)[:protein_length]
        #print("type:", type(valid_mask))
        #print("valid_mask", valid_mask)
        contact_map = np.less(squareform(pdist(torch.tensor(item['tertiary']))), 8.0).astype(np.int64)
        contact_map = contact_map[:protein_length, :protein_length]
        
        yind, xind = np.indices(contact_map.shape)
        # DEL
        invalid_mask = ~(valid_mask[:, None] & valid_mask[None, :])
        invalid_mask |= np.abs(yind - xind) < 6
        contact_map[invalid_mask] = -1

        if not self.need_features:
            return token_ids, protein_length, input_mask, contact_map
        else:
            return token_ids, protein_length, input_mask, contact_map, self.features[index]
    
    def collate_fn(self, batch):
        if not self.need_features:
            input_ids, protein_length, input_mask, contact_labels = tuple(zip(*batch))
        else:
            input_ids, protein_length, input_mask, contact_labels, features = tuple(zip(*batch))

        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        contact_labels = torch.from_numpy(pad_sequences(contact_labels, -1))
        protein_length = torch.LongTensor(protein_length)  # type: ignore
        

        result =  {'input_ids': input_ids,
                'attention_mask': input_mask,
                'labels': contact_labels,
                'protein_length': protein_length}
        
        if self.need_features:
            features = torch.from_numpy(np.array(features))
            result['features'] = features
            
        return result


class FluorescenceDataset(Dataset):
    def __init__(self, file_path, split, max_len, tokenizer,
                 need_features=False,
                 preprocess_device='0',
                 feature_type='esm'):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.max_len = max_len
        self.feature_type = feature_type

        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test'")

        data_file = f'{self.file_path}/fluorescence/fluorescence_{split}.json'
        self.seqs, self.labels = self.get_data(data_file)
        
        self.need_features = need_features
        if self.need_features:
            store_path = f'{self.file_path}/fluorescence/fluorescence_{split}.fvecs'
            self.features = get_features(self.seqs, store_path, preprocess_device, feature_type=self.feature_type)

    def get_data(self, file):
        # print(file)
        fp = pd.read_json(file)
        seqs = fp.primary
        labels = fp.log_fluorescence

        return seqs, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        seq = list(re.sub(r"[UZOB]", "X", self.seqs[index]))

        input_ids = self.tokenizer(seq, is_split_into_words=True, truncation=True, padding="max_length", max_length=self.max_len)
        input_ids = np.array(input_ids['input_ids'])
        input_mask = np.ones_like(input_ids)

        label = self.labels[index]
        if not self.need_features:
            return input_ids, input_mask, label
        else:
            return input_ids, input_mask, label, self.features[index]

    def collate_fn(self, batch):
        if not self.need_features:
            input_ids, input_mask, fluorescence_true_value = tuple(zip(*batch))
        else:
            input_ids, input_mask, fluorescence_true_value, features = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        fluorescence_true_value = torch.FloatTensor(fluorescence_true_value)  # type: ignore

        #print(fluorescence_true_value.shape)
        result =  {'input_ids': input_ids,
                'attention_mask': input_mask,
                'labels': fluorescence_true_value}
        if self.need_features:
            features = torch.from_numpy(np.array(features))
            result['features'] = features
        return result

class StabilityDataset(Dataset):
    def __init__(self, file_path, split, max_len, tokenizer, need_features=False,
                 preprocess_device='0', feature_type='esm'):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test'")

        data_file = f'{self.file_path}/stability/stability_{split}.json'
        self.seqs, self.labels = self.get_data(data_file)
        
        self.feature_type = feature_type
        
        self.need_features = need_features
        if self.need_features:
            store_path = f'{self.file_path}/stability/stability_{split}.fvecs'
            self.features = get_features(self.seqs, store_path, preprocess_device, feature_type=self.feature_type)

    def get_data(self, path):
        read_file = pd.read_json(path)

        seqs = read_file.primary
        labels = read_file.stability_score

        return seqs, labels

    def __getitem__(self, index):
        seq = list(re.sub(r"[UZOB]", "X", self.seqs[index]))

        input_ids = self.tokenizer(seq, is_split_into_words=True, padding="max_length", max_length=self.max_len, truncation=True)
        input_ids = np.array(input_ids['input_ids'])
        input_mask = np.ones_like(input_ids)

        label = self.labels[index]

        if not self.need_features:
            return input_ids, input_mask, label
        else:
            return input_ids, input_mask, label, self.features[index]
        
    def __len__(self):
        return len(self.labels)

    def collate_fn(self, batch):
        if not self.need_features:
            input_ids, input_mask, stability_true_value = tuple(zip(*batch))
        else:
            input_ids, input_mask, stability_true_value, features = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        stability_true_value = torch.FloatTensor(stability_true_value)  # type: ignore

        result= {'input_ids': input_ids,
                'attention_mask': input_mask,
                'labels': stability_true_value}
        if self.need_features:
            features = torch.from_numpy(np.array(features))
            result['features'] = features
        return result

class RemoteHomologyDataset(Dataset):
    def __init__(self, file_path, split, max_len, tokenizer, need_features=False,
                 preprocess_device='0', feature_type='esm'):
        self.tokenizer = tokenizer
        self.file_path = file_path

        if split not in ('train', 'valid', 'test_fold_holdout',
                         'test_family_holdout', 'test_superfamily_holdout'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test_fold_holdout', "
                             f"'test_family_holdout', 'test_superfamily_holdout']")

        data_file = f'{self.file_path}/remote_homology/remote_homology_{split}.json'
        self.max_len = max_len
        self.seqs, self.labels = self.get_data(data_file)
        
        self.feature_type = feature_type
        
        self.need_features = need_features
        if self.need_features:
            store_path = f'{self.file_path}/remote_homology/remote_homology_{split}.fvecs'
            self.features = get_features(self.seqs, store_path, preprocess_device, feature_type=self.feature_type)

    def get_data(self, file):
        fp = pd.read_json(file)

        seqs = fp.primary
        labels = fp.fold_label

        return seqs, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        seq = list(re.sub(r"[UZOB]", "X", self.seqs[index]))

        input_ids = self.tokenizer(seq, is_split_into_words=True, truncation=True, padding="max_length", max_length=self.max_len)
        input_ids = np.array(input_ids['input_ids'])
        input_mask = np.ones_like(input_ids)

        label = self.labels[index]

        if not self.need_features:
            return input_ids, input_mask, label
        else:
            return input_ids, input_mask, label, self.features[index]

    def collate_fn(self, batch):
        if not self.need_features:
            input_ids, input_mask, fold_label = tuple(zip(*batch))
        else:
            input_ids, input_mask, fold_label, features= tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        fold_label = torch.LongTensor(fold_label)  # type: ignore

        result =  {'input_ids': input_ids,
                'attention_mask': input_mask,
                'labels': fold_label}
        if self.need_features:
            features = torch.from_numpy(np.array(features))
            result['features'] = features
            
        return result

class SubcellularLocalizationDataset(Dataset):
    def __init__(self, file_path, split, max_len, tokenizer, need_features=False,
                 preprocess_device='0', feature_type='esm'):
        self.tokenizer = tokenizer
        self.file_path = file_path

        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test', ")

        data_file = f'{self.file_path}/subcellular_localization/subcellular_localization_{split}.lmdb'
        self.max_len = max_len
        self.data = dataset_factory(data_file)
        self.seqs = [item['primary'] for item in self.data]
        self.labels = [item['localization'] for item in self.data]
        self.feature_type = feature_type
        
        self.need_features = need_features
        if self.need_features:
            store_path = f'{self.file_path}/subcellular_localization/subcellular_localization_{split}.fvecs'
            self.features = get_features(self.seqs, store_path, preprocess_device, feature_type=self.feature_type)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        seq = list(re.sub(r"[UZOB]", "X", self.seqs[index]))

        input_ids = self.tokenizer(seq, is_split_into_words=True, truncation=True, padding="max_length", max_length=self.max_len)
        input_ids = np.array(input_ids['input_ids'])
        input_mask = np.ones_like(input_ids)

        label = self.labels[index]

        if not self.need_features:
            return input_ids, input_mask, label
        else:
            return input_ids, input_mask, label, self.features[index]

    def collate_fn(self, batch):
        if not self.need_features:
            input_ids, input_mask, label = tuple(zip(*batch))
        else:
            input_ids, input_mask, label, features= tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        label = torch.LongTensor(label)  # type: ignore

        result =  {'input_ids': input_ids,
                'attention_mask': input_mask,
                'labels': label}
        if self.need_features:
            features = torch.from_numpy(np.array(features))
            result['features'] = features
            
        return result


class SecondaryStructureDataset3(Dataset):
    def __init__(
            self,
            data_path,
            split,
            max_len, 
            tokenizer,
            in_memory,
            target='ss3',
            need_features=False,
            preprocess_device='0',
            feature_type='esm'
    ):
        self.tokenizer = tokenizer
        data_file = f'secondary_structure/secondary_structure_{split}.lmdb'
        self.data = LMDBDataset(data_file=os.path.join(data_path, data_file), in_memory=in_memory)
        self.target = target
        self.max_len = max_len
        self.ignore_index: int = -100
        self.feature_type = feature_type 
        
        self.need_features = need_features
        
        if self.need_features:
            seqs = [item['primary'] for item in self.data]
            store_path = data_path +'/'+ f'secondary_structure/secondary_structure_{split}.fvecs'
            self.features = get_features(seqs, store_path, preprocess_device, feature_type=self.feature_type)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        if len(item['primary']) > self.max_len:
            item['primary'] = item['primary'][:self.max_len] # str(protein_length)
            item['ss3'] = item['ss3'][:self.max_len] #str(protein_length)
        token_ids = self.tokenizer(list(item['primary']), is_split_into_words=True, truncation=True) # list(protein_length+2)
        token_ids = np.array(token_ids['input_ids'])
        input_mask = np.ones_like(token_ids)

        # pad with -1s because of cls/sep tokens
        labels = np.asarray(item['ss3'], np.int64)
        labels = np.pad(labels, (1, 1), 'constant', constant_values=self.ignore_index) # list(protein_length+2)

        if not self.need_features:
            return token_ids, input_mask, labels
        else:
            return token_ids, input_mask, labels, self.features[index]

    def collate_fn(self, batch):
        if not self.need_features:
            input_ids, input_mask, ss_label = tuple(zip(*batch))
        else:
            input_ids, input_mask, ss_label, features = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, constant_value=self.tokenizer.pad_token_id))
        attention_mask = torch.from_numpy(pad_sequences(input_mask, constant_value=0))
        labels = torch.from_numpy(pad_sequences(ss_label, constant_value=self.ignore_index))

        output = {'input_ids': input_ids,
                  'attention_mask': attention_mask,
                  'labels': labels}
        if self.need_features:
            features = torch.from_numpy(np.array(features))
            output['features'] = features
        return output


class SecondaryStructureDataset8(Dataset):
    def __init__(
            self,
            data_path,
            split,
            max_len,
            tokenizer,
            in_memory,
            target='ss8',
            need_features=False,
            preprocess_device='0',
            feature_type='esm'
    ):
        self.tokenizer = tokenizer
        data_file = f'secondary_structure/secondary_structure_{split}.lmdb'
        self.data = LMDBDataset(data_file=os.path.join(data_path, data_file), in_memory=in_memory)
        self.target = target
        self.max_len = max_len
        self.ignore_index: int = -100

        self.need_features = need_features
        self.feature_type = feature_type
        
        if self.need_features:
            seqs = [item['primary'] for item in self.data]
            store_path = data_path +'/'+f'secondary_structure/secondary_structure_{split}.fvecs'
            self.features = get_features(seqs, store_path, preprocess_device, feature_type=self.feature_type)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        if len(item['primary']) > self.max_len:
            item['primary'] = item['primary'][:self.max_len]
            item['ss8'] = item['ss8'][:self.max_len]
        token_ids = self.tokenizer(list(item['primary']), is_split_into_words=True, truncation=True) # list(protein_length+2)
        token_ids = np.array(token_ids['input_ids'])
        input_mask = np.ones_like(token_ids)

        # pad with -1s because of cls/sep tokens
        labels = np.asarray(item['ss8'], np.int64)
        labels = np.pad(labels, (1, 1), 'constant', constant_values=self.ignore_index)

        
        if not self.need_features:
            return token_ids, input_mask, labels
        else:
            return token_ids, input_mask, labels, self.features[index]

    def collate_fn(self, batch):
        if not self.need_features:
            input_ids, input_mask, ss_label = tuple(zip(*batch))
        else:
            input_ids, input_mask, ss_label, features = tuple(zip(*batch))

        input_ids = torch.from_numpy(pad_sequences(input_ids, constant_value=self.tokenizer.pad_token_id))
        attention_mask = torch.from_numpy(pad_sequences(input_mask, constant_value=0))
        labels = torch.from_numpy(pad_sequences(ss_label, constant_value=self.ignore_index))

        output = {'input_ids': input_ids,
                  'attention_mask': attention_mask,
                  'labels': labels}
        if self.need_features:
            features = torch.from_numpy(np.array(features))
            output['features'] = features
        return output


class ProteinProteinInteractionDataset(Dataset):
    def __init__(self, file_path, split, max_len, tokenizer, need_features=False,
                 preprocess_device='0', feature_type='esm'):
        self.tokenizer = tokenizer
        self.file_path = file_path

        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test', ")

        data_file = f'{self.file_path}/human_ppi/human_ppi_{split}.lmdb'
        self.max_len = max_len
        self.data = dataset_factory(data_file)
        self.seqs_1 = [item['primary_1'] for item in self.data]
        self.seqs_2 = [item['primary_2'] for item in self.data]
        self.labels = [item['interaction'] for item in self.data]
        self.feature_type = feature_type
        
        self.need_features = need_features
        if self.need_features:
            store_path_1 = f'{self.file_path}/human_ppi/human_ppi_{split}_1.fvecs'
            self.features_1 = get_features(self.seqs_1, store_path_1, preprocess_device, feature_type=self.feature_type)
            store_path_2 = f'{self.file_path}/human_ppi/human_ppi_{split}_2.fvecs'
            self.features_2 = get_features(self.seqs_2, store_path_2, preprocess_device, feature_type=self.feature_type)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        seq_1 = list(re.sub(r"[UZOB]", "X", self.seqs_1[index]))
        seq_2 = list(re.sub(r"[UZOB]", "X", self.seqs_2[index]))

        input_ids_1 = self.tokenizer(seq_1, is_split_into_words=True, truncation=True, padding="max_length", max_length=self.max_len)
        input_ids_1 = np.array(input_ids_1['input_ids'])
        input_mask_1 = np.ones_like(input_ids_1)

        input_ids_2 = self.tokenizer(seq_2, is_split_into_words=True, truncation=True, padding="max_length", max_length=self.max_len)
        input_ids_2 = np.array(input_ids_2['input_ids'])
        input_mask_2 = np.ones_like(input_ids_2)
        
        label = self.labels[index]

        if not self.need_features:
            return input_ids_1, input_ids_2, input_mask_1, input_mask_2, label
        else:
            return input_ids_1, input_ids_2, input_mask_1, input_mask_2, label, self.features_1[index], self.features_2[index]

    def collate_fn(self, batch):
        if not self.need_features:
            input_ids_1, input_ids_2, input_mask_1, input_mask_2, label = tuple(zip(*batch))
        else:
            input_ids_1, input_ids_2, input_mask_1, input_mask_2, label, features_1, features_2 = tuple(zip(*batch))
        input_ids_1 = torch.from_numpy(pad_sequences(input_ids_1, 0))
        input_mask_1 = torch.from_numpy(pad_sequences(input_mask_1, 0))
        input_ids_2 = torch.from_numpy(pad_sequences(input_ids_2, 0))
        input_mask_2 = torch.from_numpy(pad_sequences(input_mask_2, 0))
        label = torch.LongTensor(label)  # type: ignore

        result =  {'input_ids_1': input_ids_1,
                'attention_mask_1': input_mask_1,
                'input_ids_2': input_ids_2,
                'attention_mask_2': input_mask_2,
                'labels': label}
        if self.need_features:
            features_1 = torch.from_numpy(np.array(features_1))
            features_2 = torch.from_numpy(np.array(features_2))
            result['features_1'] = features_1
            result['features_2'] = features_2
            
        return result


output_modes_mapping = {
    'contact': 'token-level-classification',
    'remote_homology': 'sequence-level-classification',
    'fluorescence': 'sequence-level-regression',
    'stability': 'sequence-level-regression',
    'ss3': 'token-level-classification',
    'ss8': 'token-level-classification',
    'subcellular': 'sequence-level-regression',
    'ppi': 'sequence-level-regression'
}



dataset_mapping = {'bert':{
    'remote_homology': RemoteHomologyProgress,
    'fluorescence': FluorescenceProgress,
    'stability': StabilityProgress,
    'contact': ContactProgress,
    'ss3': SecondaryStructureProcessor3,
    'ss8': SecondaryStructureProcessor8,
    'subcellular': SubcellularLocalizationProgress,
    'ppi': ProteinProteinInteractionProgress
},
'lstm':{
    'remote_homology': RemoteHomologyProgress,
    'fluorescence': FluorescenceProgress,
    'stability': StabilityProgress,
    'contact': ContactProgress,
    'ss3': SecondaryStructureProcessor3,
    'ss8': SecondaryStructureProcessor8,
    'subcellular': SubcellularLocalizationProgress,
    'ppi': ProteinProteinInteractionProgress
},
'resnet':{
    'remote_homology': RemoteHomologyProgress,
    'fluorescence': FluorescenceProgress,
    'stability': StabilityProgress,
    'contact': ContactProgress,
    'ss3': SecondaryStructureProcessor3,
    'ss8': SecondaryStructureProcessor8,
    'subcellular': SubcellularLocalizationProgress,
    'ppi': ProteinProteinInteractionProgress
},
'esm':{
    'remote_homology': RemoteHomologyProgress,
    'fluorescence': FluorescenceProgress,
    'stability': StabilityProgress,
    'contact': ContactProgress,
    'ss3': SecondaryStructureProcessor3,
    'ss8': SecondaryStructureProcessor8,
    'subcellular': SubcellularLocalizationProgress,
    'ppi': ProteinProteinInteractionProgress
},
'msa':{
    'remote_homology': MSARemoteHomologyProgress,
    'fluorescence': MSAFluorescenceProgress,
    'stability': MSAStabilityProgress,
    'contact': MSAContactProgress,
    'ss3': MSASecondaryStructureProcessor3,
    'ss8': MSASecondaryStructureProcessor8,
    'subcellular': MSASubcellularLocalizationProgress,
    'ppi': MSAProteinProteinInteractionProgress
},
'augment':{
    'remote_homology': AugmentRemoteHomologyProgress,
    'fluorescence': AugmentFluorescenceProgress,
    'stability': AugmentStabilityProgress,
    'contact': AugmentContactProgress,
    'ss3': AugmentSecondaryStructureProcessor3,
    'ss8': AugmentSecondaryStructureProcessor8,
    'subcellular': SubcellularLocalizationProgress,
    'ppi': ProteinProteinInteractionProgress
},

}