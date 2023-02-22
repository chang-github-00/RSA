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

def first_select(msa,num_seqs):
    if len(msa) <= num_seqs:
        return msa
    else:
        return msa[0:num_seqs]

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


class MSAFluorescenceProgress(DataProcessor):
    def __init__(self, max_len, tokenizer, need_features=False, preprocess_device='0',select_method='random',num_msa=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device
        self.select_method=select_method
        self.num_msa=num_msa

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = MSAFluorescenceDataset(data_dir, split='train', max_len=self.max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = MSAFluorescenceDataset(data_dir, split='valid', max_len=self.max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                      need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        if data_cat is not None:
            dataset = MSAFluorescenceDataset(data_dir, split=data_cat, max_len=self.max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                          need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        else:
            dataset = MSAFluorescenceDataset(data_dir, split='test', max_len=self.max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                          need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_labels(self):
        return list(range(1))


class MSASecondaryStructureProcessor3(DataProcessor):
    def __init__(self, max_len, tokenizer, need_features=False, preprocess_device='0',select_method='random',num_msa=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device
        self.select_method=select_method
        self.num_msa=num_msa

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = MSASecondaryStructureDataset3(data_dir, split='train', max_len=self.max_len, tokenizer=self.tokenizer,
                                             target='ss3', in_memory=in_memory,
                                             need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = MSASecondaryStructureDataset3(data_dir, split='valid', max_len=self.max_len, tokenizer=self.tokenizer,
                                             target='ss3', in_memory=in_memory,
                                             need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        dataset = MSASecondaryStructureDataset3(data_dir, split=data_cat, max_len=self.max_len, tokenizer=self.tokenizer,
                                             target='ss3', in_memory=in_memory,
                                             need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_labels(self):
        return list(range(3))


class MSASecondaryStructureProcessor8(DataProcessor):
    def __init__(self, max_len, tokenizer, need_features=False, preprocess_device='0',select_method='random',num_msa=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device
        self.select_method=select_method
        self.num_msa=num_msa

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = MSASecondaryStructureDataset8(data_dir, split='train', max_len=self.max_len, tokenizer=self.tokenizer,
                                             target='ss8', in_memory=in_memory,
                                             need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = MSASecondaryStructureDataset8(data_dir, split='valid', max_len=self.max_len, tokenizer=self.tokenizer,
                                             target='ss8', in_memory=in_memory,
                                             need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        dataset = MSASecondaryStructureDataset8(data_dir, split=data_cat, max_len=self.max_len, tokenizer=self.tokenizer,
                                             target='ss8', in_memory=in_memory,
                                             need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_labels(self):
        return list(range(8))


class MSAContactProgress(DataProcessor):
    def __init__(self, max_len, tokenizer, need_features=False, preprocess_device='0',select_method='random',num_msa=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device
        self.select_method=select_method
        self.num_msa=num_msa

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = MSAProteinnetDataset(data_dir, split='train', max_len=self.max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                    need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = MSAProteinnetDataset(data_dir, split='valid', max_len=self.max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                    need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        if data_cat is not None:
            dataset = MSAProteinnetDataset(data_dir, split=data_cat, max_len=self.max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                        need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        else:
            dataset = MSAProteinnetDataset(data_dir, split='test', max_len=self.max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                        need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_labels(self):
        return list(range(2))


class MSAStabilityProgress(DataProcessor):
    def __init__(self, max_len, tokenizer, need_features=False, preprocess_device='0',select_method='random',num_msa=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device
        self.select_method=select_method
        self.num_msa=num_msa

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = MSAStabilityDataset(data_dir, split='train', max_len=self.max_len, tokenizer=self.tokenizer, in_memory=in_memory,
                                   need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = MSAStabilityDataset(data_dir, split='valid', max_len=self.max_len, tokenizer=self.tokenizer, in_memory=in_memory,
                                   need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        if data_cat is not None:
            dataset = MSAStabilityDataset(data_dir, split=data_cat, max_len=self.max_len, tokenizer=self.tokenizer, in_memory=in_memory,
                                       need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        else:
            dataset = MSAStabilityDataset(data_dir, split='test', max_len=self.max_len, tokenizer=self.tokenizer, in_memory=in_memory,
                                       need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_labels(self):
        return list(range(1))


class MSARemoteHomologyProgress(DataProcessor):
    def __init__(self, max_len, tokenizer, need_features=False, preprocess_device='0',select_method='random',num_msa=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device
        self.select_method=select_method
        self.num_msa=num_msa

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = MSARemoteHomologyDataset(data_dir, split='train', max_len=self.max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                        need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = MSARemoteHomologyDataset(data_dir, split='valid', max_len=self.max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                        need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        if data_cat is not None:
            dataset = MSARemoteHomologyDataset(data_dir, split=data_cat, max_len=self.max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                            need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        else:
            dataset = MSARemoteHomologyDataset(data_dir, split='test', max_len=self.max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                            need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_labels(self):
        return list(range(1195))


class MSAProteinProteinInteractionProgress(DataProcessor):
    def __init__(self, max_len, tokenizer, need_features=False, preprocess_device='0',select_method='random',num_msa=128, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device
        self.select_method=select_method
        self.num_msa=num_msa

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = MSAProteinProteinInteractionDataset(data_dir, split='train', max_len=self.max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                            need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = MSAProteinProteinInteractionDataset(data_dir, split='valid', max_len=self.max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                            need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        if data_cat is not None:
            dataset = MSAProteinProteinInteractionDataset(data_dir, split=data_cat, max_len=self.max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                            need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        else:
            dataset = MSAProteinProteinInteractionDataset(data_dir, split='test', max_len=self.max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                            need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_labels(self):
        return list(range(2))


class MSASubcellularLocalizationProgress(DataProcessor):
    def __init__(self, max_len, tokenizer, need_features=False, preprocess_device='0',select_method='random',num_msa=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.need_features = need_features
        self.prepocess_device = preprocess_device
        self.select_method=select_method
        self.num_msa=num_msa

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = MSASubcellularLocalizationDataset(data_dir, split='train', max_len=self.max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                        need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = MSASubcellularLocalizationDataset(data_dir, split='valid', max_len=self.max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                        need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        if data_cat is not None:
            dataset = MSASubcellularLocalizationDataset(data_dir, split=data_cat, max_len=self.max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                            need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        else:
            dataset = MSASubcellularLocalizationDataset(data_dir, split='test', max_len=self.max_len, tokenizer=self.tokenizer,in_memory=in_memory,
                                            need_features=self.need_features, preprocess_device=self.prepocess_device,select_method=self.select_method,num_msa=self.num_msa)
        return dataset

    def get_labels(self):
        return list(range(1195))






class MSAProteinnetDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 max_len: int,
                 tokenizer,
                 in_memory,
                 need_features=False,
                 preprocess_device='0',
                 select_method='random',
                 num_msa=10
                 ):

        if split not in ('train', 'train_unfiltered', 'valid', 'test', 'de_novo'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'train_unfiltered', 'valid', 'test']")

        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'proteinnet/proteinnet_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory=in_memory)
        self.max_len = max_len
        self.need_features = need_features

        if self.need_features:
            seqs = [item['primary'] for item in self.data]
            store_path = str(data_path / f'proteinnet/proteinnet_{split}.fvecs')
            self.features = get_features(seqs, store_path, preprocess_device)

        self.select_fn = {'random': random_select, 'greedy': greedy_select,'first':first_select}[select_method]
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

        max_len = self.max_len
        if 'msa' not in item:
            item['msa'] = []
        msa_lists = [item['primary']] + item['msa']
        input_paded = []
        for data in msa_lists:
            data=re.sub(r"[a-zUZOB]", "X", data)
            if len(data) < self.max_len:
                # self.tokenizer.msa_transformer_batch_converter.alphabet.append_toks[0]
                # pad = '.' * (self.max_len - len(data))
                pad=''
                input_paded.append(('', data + pad))
            else:
                input_paded.append(('', data[0:self.max_len]))

        selected = self.select_fn(input_paded, num_seqs=self.num_msa)

        msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens  = self.tokenizer(selected,append_eos=True)  # list(protein_length+2)
        token_ids = np.array(msa_transformer_batch_tokens)
        # input_mask = None

        seq = list(re.sub(r"[UZOB]", "X", item['primary']))
        protein_length = min(len(seq), max_len)

        # seq = seq[:protein_length]

        # token_ids = self.tokenizer(seq, is_split_into_words=True)
        # token_ids = np.asarray(token_ids['input_ids'], dtype=int)

        input_mask = np.ones([1,protein_length+2])

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

        contact_map = contact_map.reshape([1, protein_length, protein_length])

        if not self.need_features:
            return token_ids, protein_length, input_mask, contact_map
        else:
            return token_ids, protein_length, input_mask, contact_map, self.features[index]

    def collate_fn(self, batch):
        if not self.need_features:
            input_ids, protein_length, input_mask, contact_labels = tuple(zip(*batch))
        else:
            input_ids, protein_length, input_mask, contact_labels, features = tuple(zip(*batch))

        input_ids = torch.from_numpy(np.concatenate(input_ids,0))
        input_mask = torch.from_numpy(np.concatenate(input_mask,0))
        contact_labels = torch.from_numpy(np.concatenate(contact_labels,0))
        protein_length = torch.LongTensor(protein_length)  # type: ignore

        result = {'input_ids': input_ids,
                  'attention_mask': input_mask,
                  'labels': contact_labels,
                  'protein_length': protein_length}

        # if self.need_features:
        #     features = torch.from_numpy(np.array(features))
        #     result['features'] = features
        # if not self.need_features:
        #     input_ids, input_mask, ss_label = tuple(zip(*batch))
        # else:
        #     input_ids, input_mask, ss_label, features = tuple(zip(*batch))
        # input_ids = torch.from_numpy(np.concatenate(input_ids,0))
        # attention_mask = None
        # labels = torch.from_numpy(np.concatenate(ss_label,0))
        #
        # output = {'input_ids': input_ids,
        #           'attention_mask': attention_mask,
        #           'labels': labels}

        return result


class MSAFluorescenceDataset(Dataset):
    def __init__(self, data_path, split, max_len, tokenizer,in_memory,
                 need_features=False,
                 preprocess_device='0',
                 select_method='random',
                 num_msa=10):
        self.tokenizer = tokenizer
        self.max_len = max_len

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


        self.select_fn = {'random': random_select, 'greedy': greedy_select,'first':first_select}[select_method]
        self.num_msa = num_msa


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.data[index]
        except:
            return self.__getitem__(np.random.randint(len(self.data)))

        msa_lists = [item['primary']] + item['msa']
        input_paded = []
        for data in msa_lists:
            data=re.sub(r"[a-zUZOB]", "X", data)
            if len(data) < self.max_len:
                # self.tokenizer.msa_transformer_batch_converter.alphabet.append_toks[0]
                # pad = '.' * (self.max_len - len(data))
                pad=''
                input_paded.append(('', data + pad))
            else:
                input_paded.append(('', data[0:self.max_len]))

        selected = self.select_fn(input_paded, num_seqs=self.num_msa)
        # item['selected'] = selected

        msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens  = self.tokenizer(selected)  # list(protein_length+2)
        token_ids = np.array(msa_transformer_batch_tokens)
        input_mask = None

        # pad with -1s because of cls/sep tokens
        labels = np.asarray(item['log_fluorescence'], np.float32)


        if not self.need_features:
            return token_ids, input_mask, labels
        else:
            return token_ids, input_mask, labels, self.features[index]



    def collate_fn(self, batch):
        if not self.need_features:
            input_ids, input_mask, ss_label = tuple(zip(*batch))
        else:
            input_ids, input_mask, ss_label, features = tuple(zip(*batch))
        input_ids = torch.from_numpy(np.concatenate(input_ids,0))
        attention_mask = None
        labels = torch.from_numpy(np.concatenate(ss_label,0))

        output = {'input_ids': input_ids,
                  'attention_mask': attention_mask,
                  'labels': labels}

        return output


class MSAStabilityDataset(Dataset):
    def __init__(self, data_path, split, max_len, tokenizer,in_memory,
                 need_features=False,
                 preprocess_device='0',
                 select_method='random',
                 num_msa=10):
        self.tokenizer = tokenizer
        self.max_len = max_len

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

        msa_lists = [item['primary']] + item['msa']
        input_paded = []
        for data in msa_lists:
            data=re.sub(r"[a-zUZOB]", "X", data)
            if len(data) < self.max_len:
                # self.tokenizer.msa_transformer_batch_converter.alphabet.append_toks[0]
                # pad = '.' * (self.max_len - len(data))
                pad=''
                input_paded.append(('', data + pad))
            else:
                input_paded.append(('', data[0:self.max_len]))

        selected = self.select_fn(input_paded, num_seqs=self.num_msa)
        # item['selected'] = selected

        msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens  = self.tokenizer(selected)  # list(protein_length+2)
        token_ids = np.array(msa_transformer_batch_tokens)
        input_mask = None

        # pad with -1s because of cls/sep tokens
        labels = np.asarray(item['stability_score'], np.float32)


        if not self.need_features:
            return token_ids, input_mask, labels
        else:
            return token_ids, input_mask, labels, self.features[index]



    def collate_fn(self, batch):
        if not self.need_features:
            input_ids, input_mask, ss_label = tuple(zip(*batch))
        else:
            input_ids, input_mask, ss_label, features = tuple(zip(*batch))
        input_ids = torch.from_numpy(np.concatenate(input_ids,0))
        attention_mask = None
        labels = torch.from_numpy(np.concatenate(ss_label,0))

        output = {'input_ids': input_ids,
                  'attention_mask': attention_mask,
                  'labels': labels}

        return output



class MSARemoteHomologyDataset(Dataset):
    def __init__(self, data_path, split, max_len, tokenizer,in_memory,
                 need_features=False,
                 preprocess_device='0',
                 select_method='random',
                 num_msa=10):
        self.tokenizer = tokenizer
        self.max_len = max_len


        if split not in ('train', 'valid', 'test_fold_holdout',
                         'test_family_holdout', 'test_superfamily_holdout'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test_fold_holdout', "
                             f"'test_family_holdout', 'test_superfamily_holdout']")


        data_file = os.path.join('remote_homology',f'remote_homology_{split}.lmdb')
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

        msa_lists = [item['primary']] + item['msa']
        input_paded = []
        for data in msa_lists:
            data=re.sub(r"[a-zUZOB]", "X", data)
            if len(data) < self.max_len:
                # self.tokenizer.msa_transformer_batch_converter.alphabet.append_toks[0]
                # pad = '.' * (self.max_len - len(data))
                pad=''
                input_paded.append(('', data + pad))
            else:
                input_paded.append(('', data[0:self.max_len]))

        selected = self.select_fn(input_paded, num_seqs=self.num_msa)
        # item['selected'] = selected

        msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens  = self.tokenizer(selected)  # list(protein_length+2)
        token_ids = np.array(msa_transformer_batch_tokens)
        input_mask = None

        # pad with -1s because of cls/sep tokens
        labels = np.asarray([item['fold_label']], np.int64)


        if not self.need_features:
            return token_ids, input_mask, labels
        else:
            return token_ids, input_mask, labels, self.features[index]



    def collate_fn(self, batch):
        if not self.need_features:
            input_ids, input_mask, ss_label = tuple(zip(*batch))
        else:
            input_ids, input_mask, ss_label, features = tuple(zip(*batch))
        input_ids = torch.from_numpy(np.concatenate(input_ids,0))
        attention_mask = None
        labels = torch.from_numpy(np.concatenate(ss_label,0))

        output = {'input_ids': input_ids,
                  'attention_mask': attention_mask,
                  'labels': labels}

        return output





class MSASecondaryStructureDataset3(Dataset):
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
            select_method='random',
            num_msa=10
    ):
        self.tokenizer = tokenizer
        data_file = f'secondary_structure/secondary_structure_{split}.lmdb'
        self.data = LMDBDataset(data_file=os.path.join(data_path, data_file), in_memory=in_memory)

        self.select_fn={'random':random_select,'greedy':greedy_select,'first':first_select}[select_method]
        self.target = target
        self.max_len = max_len
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
        msa_lists = [item['primary']] + item['msa']
        input_paded = []
        for data in msa_lists:
            data=re.sub(r"[a-zUZOB]", "X", data)
            if len(data) < self.max_len:
                # self.tokenizer.msa_transformer_batch_converter.alphabet.append_toks[0]
                # pad = '.' * (self.max_len - len(data))
                pad=''
                input_paded.append(('', data + pad))
            else:
                input_paded.append(('', data[0:self.max_len]))

        selected = self.select_fn(input_paded, num_seqs=self.num_msa)
        # item['selected'] = selected


        if len(item['primary']) > self.max_len:
            item['primary'] = item['primary'][:self.max_len]  # str(protein_length)
            item['ss3'] = item['ss3'][:self.max_len]  # str(protein_length)

        msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens  = self.tokenizer(selected)  # list(protein_length+2)
        token_ids = np.array(msa_transformer_batch_tokens)
        input_mask = None

        # pad with -1s because of cls/sep tokens
        labels = np.asarray(item['ss3'], np.int64)
        labels = np.pad(labels, (1, token_ids.shape[-1]-len(labels)-1), 'constant', constant_values=self.ignore_index)  # list(protein_length+2)
        labels=np.reshape(labels,[1,-1])

        if not self.need_features:
            return token_ids, input_mask, labels
        else:
            return token_ids, input_mask, labels, self.features[index]

    def collate_fn(self, batch):
        if not self.need_features:
            input_ids, input_mask, ss_label = tuple(zip(*batch))
        else:
            input_ids, input_mask, ss_label, features = tuple(zip(*batch))
        input_ids = torch.from_numpy(np.concatenate(input_ids,0))
        attention_mask = None
        labels = torch.from_numpy(np.concatenate(ss_label,0))

        output = {'input_ids': input_ids,
                  'attention_mask': attention_mask,
                  'labels': labels}
        if self.need_features:
            features = torch.from_numpy(np.array(features))
            output['features'] = features
        return output


class MSASecondaryStructureDataset8(Dataset):
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
            select_method='random',
            num_msa=10

    ):
        self.tokenizer = tokenizer
        data_file = f'secondary_structure/secondary_structure_{split}.lmdb'
        self.data = LMDBDataset(data_file=os.path.join(data_path, data_file), in_memory=in_memory)
        self.target = target
        self.max_len = max_len
        self.ignore_index: int = -100

        self.need_features = need_features

        if self.need_features:
            seqs = [item['primary'] for item in self.data]
            store_path = data_path + '/' + f'secondary_structure/secondary_structure_{split}.fvecs'
            self.features = get_features(seqs, store_path, preprocess_device)

        self.select_fn = {'random': random_select, 'greedy': greedy_select,'first':first_select}[select_method]
        self.num_msa = num_msa

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        try:
            item = self.data[index]
        except:
            return self.__getitem__(np.random.randint(len(self.data)))

        #msa selecting
        if 'msa' not in item:
            item['msa'] = []
        msa_lists = [item['primary']] + item['msa']
        input_paded = []
        for data in msa_lists:
            data=re.sub(r"[a-zUZOB]", "X", data)
            if len(data) < self.max_len:
                # self.tokenizer.msa_transformer_batch_converter.alphabet.append_toks[0]
                # pad = '.' * (self.max_len - len(data))
                pad=''
                input_paded.append(('', data + pad))
            else:
                input_paded.append(('', data[0:self.max_len]))

        selected = self.select_fn(input_paded, num_seqs=self.num_msa)

        # msa tokenizing
        msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens  = self.tokenizer(selected)  # list(protein_length+2)
        token_ids = np.array(msa_transformer_batch_tokens)
        input_mask = None

        #label
        if len(item['primary']) > self.max_len:
            item['primary'] = item['primary'][:self.max_len]  # str(protein_length)
            item['ss8'] = item['ss8'][:self.max_len]  # str(protein_length)

        # pad with -1s because of cls/sep tokens
        labels = np.asarray(item['ss8'], np.int64)
        labels = np.pad(labels, (1, token_ids.shape[-1]-len(labels)-1), 'constant', constant_values=self.ignore_index)  # list(protein_length+2)
        labels=np.reshape(labels,[1,-1])

        if not self.need_features:
            return token_ids, input_mask, labels
        else:
            return token_ids, input_mask, labels, self.features[index]

    def collate_fn(self, batch):
        if not self.need_features:
            input_ids, input_mask, ss_label = tuple(zip(*batch))
        else:
            input_ids, input_mask, ss_label, features = tuple(zip(*batch))
        input_ids = torch.from_numpy(np.concatenate(input_ids,0))
        attention_mask = None
        labels = torch.from_numpy(np.concatenate(ss_label,0))

        output = {'input_ids': input_ids,
                  'attention_mask': attention_mask,
                  'labels': labels}
        if self.need_features:
            features = torch.from_numpy(np.array(features))
            output['features'] = features
        return output





class MSAProteinProteinInteractionDataset(Dataset):
    def __init__(self, data_path, split, max_len, tokenizer,in_memory, need_features=False,
                 preprocess_device='0', select_method = 'random',num_msa = 10,feature_type=None):
        self.tokenizer = tokenizer
        self.file_path = data_path

        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test', ")

        data_file = f'{self.file_path}/human_ppi/human_ppi_{split}.lmdb'
        self.max_len = max_len
        self.data = dataset_factory(data_file)
        # self.seqs_1 = [item['primary_1'] for item in self.data]
        # self.seqs_2 = [item['primary_2'] for item in self.data]
        # self.labels = [item['interaction'] for item in self.data]
        # self.feature_type = feature_type

        # self.need_features = need_features
        # if self.need_features:
        #     store_path_1 = f'{self.file_path}/human_ppi/human_ppi_{split}_1.fvecs'
        #     self.features_1 = get_features(self.seqs_1, store_path_1, preprocess_device, feature_type=self.feature_type)
        #     store_path_2 = f'{self.file_path}/human_ppi/human_ppi_{split}_2.fvecs'
        #     self.features_2 = get_features(self.seqs_2, store_path_2, preprocess_device, feature_type=self.feature_type)


        # self.tokenizer = tokenizer
        # data_file = f'secondary_structure/secondary_structure_{split}.lmdb'
        # self.data = LMDBDataset(data_file=os.path.join(data_path, data_file), in_memory=in_memory)
        # self.target = target
        # self.max_len = max_len
        # self.ignore_index: int = -100
        #
        self.need_features = need_features
        #
        # if self.need_features:
        #     seqs = [item['primary'] for item in self.data]
        #     store_path = data_path + '/' + f'secondary_structure/secondary_structure_{split}.fvecs'
        #     self.features = get_features(seqs, store_path, preprocess_device)

        self.select_fn = {'random': random_select, 'greedy': greedy_select,'first':first_select}[select_method]
        self.num_msa = num_msa





    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # seq_1 = list(re.sub(r"[UZOB]", "X", self.seqs_1[index]))
        # seq_2 = list(re.sub(r"[UZOB]", "X", self.seqs_2[index]))
        #
        # input_ids_1 = self.tokenizer(seq_1, is_split_into_words=True, truncation=True, padding="max_length",
        #                              max_length=self.max_len)
        # input_ids_1 = np.array(input_ids_1['input_ids'])
        # input_mask_1 = np.ones_like(input_ids_1)
        #
        # input_ids_2 = self.tokenizer(seq_2, is_split_into_words=True, truncation=True, padding="max_length",
        #                              max_length=self.max_len)
        # input_ids_2 = np.array(input_ids_2['input_ids'])
        # input_mask_2 = np.ones_like(input_ids_2)
        #
        # label = self.labels[index]
        #
        # if not self.need_features:
        #     return input_ids_1, input_ids_2, input_mask_1, input_mask_2, label
        # else:
        #     return input_ids_1, input_ids_2, input_mask_1, input_mask_2, label, self.features_1[index], self.features_2[
        #         index]


        # try:
        item = self.data[index]
        # except:
        #     return self.__getitem__(np.random.randint(len(self.data)))

        msa_lists_1 = [item['primary_1']] + item['msa_1']
        input_paded_1 = []
        for data in msa_lists_1:
            data=re.sub(r"[a-zUZOB]", "X", data)
            if len(data) < self.max_len:
                # self.tokenizer.msa_transformer_batch_converter.alphabet.append_toks[0]
                # pad = '.' * (self.max_len - len(data))
                pad=''
                input_paded_1.append(('', data + pad))
            else:
                input_paded_1.append(('', data[0:self.max_len]))

        selected_1 = self.select_fn(input_paded_1, num_seqs=self.num_msa)
        # item['selected'] = selected

        msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens  = self.tokenizer(selected_1)  # list(protein_length+2)
        input_ids_1 = np.array(msa_transformer_batch_tokens)
        input_mask_1 = None


        msa_lists_2 = [item['primary_2']] + item['msa_2']
        input_paded_2 = []
        for data in msa_lists_2:
            data=re.sub(r"[a-zUZOB]", "X", data)
            if len(data) < self.max_len:
                # self.tokenizer.msa_transformer_batch_converter.alphabet.append_toks[0]
                # pad = '.' * (self.max_len - len(data))
                pad=''
                input_paded_2.append(('', data + pad))
            else:
                input_paded_2.append(('', data[0:self.max_len]))

        selected_2 = self.select_fn(input_paded_2, num_seqs=self.num_msa)
        # item['selected'] = selected

        msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens  = self.tokenizer(selected_2)  # list(protein_length+2)
        input_ids_2 = np.array(msa_transformer_batch_tokens)
        input_mask_2 = None

        # pad with -1s because of cls/sep tokens
        label = np.asarray([item['interaction']], np.int64)

        if not self.need_features:
            return input_ids_1, input_ids_2, input_mask_1, input_mask_2, label
        else:
            return input_ids_1, input_ids_2, input_mask_1, input_mask_2, label, self.features_1[index], self.features_2[index]






    def collate_fn(self, batch):
        if not self.need_features:
            input_ids_1, input_ids_2, input_mask_1, input_mask_2, label = tuple(zip(*batch))
        else:
            input_ids_1, input_ids_2, input_mask_1, input_mask_2, label, features_1, features_2 = tuple(zip(*batch))
        input_ids_1 = torch.from_numpy(np.concatenate(input_ids_1, 0))
        input_mask_1 = None
        input_ids_2 = torch.from_numpy(np.concatenate(input_ids_2, 0))
        input_mask_2 = None
        labels = torch.from_numpy(np.concatenate(label,0))

        result = {'input_ids_1': input_ids_1,
                  'attention_mask_1': input_mask_1,
                  'input_ids_2': input_ids_2,
                  'attention_mask_2': input_mask_2,
                  'labels': labels}
        if self.need_features:
            features_1 = torch.from_numpy(np.array(features_1))
            features_2 = torch.from_numpy(np.array(features_2))
            result['features_1'] = features_1
            result['features_2'] = features_2

        return result


        #
        # if not self.need_features:
        #     input_ids, input_mask, ss_label = tuple(zip(*batch))
        # else:
        #     input_ids, input_mask, ss_label, features = tuple(zip(*batch))
        # input_ids = torch.from_numpy(np.concatenate(input_ids,0))
        # attention_mask = None
        # labels = torch.from_numpy(np.concatenate(ss_label,0))
        #
        # output = {'input_ids': input_ids,
        #           'attention_mask': attention_mask,
        #           'labels': labels}
        # if self.need_features:
        #     features = torch.from_numpy(np.array(features))
        #     output['features'] = features
        # return output



class MSASubcellularLocalizationDataset(Dataset):
    def __init__(self, data_path, split, max_len, tokenizer,in_memory,
                 need_features=False,
                 preprocess_device='0',
                 select_method='random',
                 num_msa=10):
        self.tokenizer = tokenizer
        self.max_len = max_len


        if split not in ('train', 'valid', 'test',
                         ):
            raise ValueError(f"Unrecognized split: {split}. Must be one of ['train', 'valid', 'test]")


        data_file = os.path.join('subcellular_localization',f'subcellular_localization_{split}.lmdb')
        self.data = LMDBDataset(data_file=os.path.join(data_path, data_file), in_memory=in_memory)
        # self.seqs, self.labels = self.get_data(data_file)

        self.need_features = need_features
        if self.need_features:
            seqs = [item['primary'] for item in self.data]
            store_path = f'{data_path}/subcellular_localization/subcellular_localization{split}.fvecs'
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

        msa_lists = [item['primary']] + item['msa']
        input_paded = []
        for data in msa_lists:
            data=re.sub(r"[a-zUZOB]", "X", data)
            if len(data) < self.max_len:
                # self.tokenizer.msa_transformer_batch_converter.alphabet.append_toks[0]
                # pad = '.' * (self.max_len - len(data))
                pad=''
                input_paded.append(('', data + pad))
            else:
                input_paded.append(('', data[0:self.max_len]))

        selected = self.select_fn(input_paded, num_seqs=self.num_msa)
        # item['selected'] = selected

        msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens  = self.tokenizer(selected)  # list(protein_length+2)
        token_ids = np.array(msa_transformer_batch_tokens)
        input_mask = None

        # pad with -1s because of cls/sep tokens
        labels = np.asarray([item['localization']], np.int64)


        if not self.need_features:
            return token_ids, input_mask, labels
        else:
            return token_ids, input_mask, labels, self.features[index]



    def collate_fn(self, batch):
        if not self.need_features:
            input_ids, input_mask, ss_label = tuple(zip(*batch))
        else:
            input_ids, input_mask, ss_label, features = tuple(zip(*batch))
        input_ids = torch.from_numpy(np.concatenate(input_ids,0))
        attention_mask = None
        labels = torch.from_numpy(np.concatenate(ss_label,0))

        output = {'input_ids': input_ids,
                  'attention_mask': attention_mask,
                  'labels': labels}

        return output




