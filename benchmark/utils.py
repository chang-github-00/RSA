import esm
import torch
import os
import numpy as np
from tqdm import tqdm


def ivecs_read(fname, count=-1, offset=0):
    a = np.fromfile(fname, dtype='int32', count=count, offset=offset)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname, count=-1, offset=0):
    return ivecs_read(fname, count=count, offset=offset).view('float32')


def ivecs_write(fname, m):
    n, d = m.shape
    # print("dimension is :"+str(d))
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    # print(m1.shape)
    m1.tofile(fname)


def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))


def get_features(seqs, store_path, preprocess_device, feature_type='esm'):
    if feature_type == 'esm':
        return get_esm_features(seqs, store_path, preprocess_device)
    if feature_type == 'esm_2':
        return get_esm_2_features(seqs, store_path, preprocess_device)
    if feature_type == 'bm25':
        raise NotImplementedError

def get_esm_features(seqs, store_path, preprocess_device): 
    if os.path.exists(store_path) and True:
        features = fvecs_read(store_path)
    else:
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        batch_converter = alphabet.get_batch_converter()

        model.eval()

        preprocess_device = 'cuda:' + str(preprocess_device) if torch.cuda.is_available() else 'cpu'

        model = model.to(preprocess_device)
        # print(store_path)
        with open(store_path, "wb") as f_out:
            for seq in tqdm(seqs, total=len(seqs)):
                sentence_batch = []
                sentence_batch.append(('seq', seq))
                batch_labels, batch_strs, batch_tokens = batch_converter(sentence_batch)
                b, n = batch_tokens.shape
                if n > 1024:
                    batch_tokens = batch_tokens[:, :1024]
                batch_tokens = batch_tokens.to(preprocess_device)
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                    token_representations = results["representations"][33]
                sequence_representations = []
                for i, (_, seq) in enumerate(sentence_batch):
                    sequence_representations.append(np.array(token_representations[i, 1: len(seq) + 1].mean(0).cpu()))
                fvecs_write(f_out, np.array(sequence_representations))

        features = fvecs_read(store_path)
    return features

def get_esm_2_features(seqs, store_path, preprocess_device): 
    if os.path.exists(store_path) and True:
        features = fvecs_read(store_path)
    else:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()

        model.eval()

        preprocess_device = 'cuda:' + str(preprocess_device) if torch.cuda.is_available() else 'cpu'

        model = model.to(preprocess_device)
        # print(store_path)
        with open(store_path, "wb") as f_out:
            for seq in tqdm(seqs, total=len(seqs)):
                sentence_batch = []
                sentence_batch.append(('seq', seq))
                batch_labels, batch_strs, batch_tokens = batch_converter(sentence_batch)
                b, n = batch_tokens.shape
                if n > 1024:
                    batch_tokens = batch_tokens[:, :1024]
                batch_tokens = batch_tokens.to(preprocess_device)
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                    token_representations = results["representations"][33]
                sequence_representations = []
                for i, (_, seq) in enumerate(sentence_batch):
                    sequence_representations.append(np.array(token_representations[i, 1: len(seq) + 1].mean(0).cpu()))
                fvecs_write(f_out, np.array(sequence_representations))

        features = fvecs_read(store_path)
    return features


def get_esm_feature(seq, model, alphabet, preprocess_device):
    batch_converter = alphabet.get_batch_converter()
    
    model.eval()
    
    
    preprocess_device = 'cuda:' + str(preprocess_device) if torch.cuda.is_available() else 'cpu'

    model = model.to(preprocess_device)
    
    sentence_batch = []
    sentence_batch.append(('seq', seq))
    batch_labels, batch_strs, batch_tokens = batch_converter(sentence_batch)
    b, n = batch_tokens.shape
    if n > 1024:
        batch_tokens = batch_tokens[:, :1024]
    batch_tokens = batch_tokens.to(preprocess_device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
    return token_representations[0, 1: len(seq) + 1].mean(0).cpu()
