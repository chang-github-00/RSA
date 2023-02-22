import faiss
import torch
import math
import numpy as np
import time
import torch.nn.functional as F

def ivecs_read(fname, count=-1, offset=0):
    a = np.fromfile(fname, dtype='int32', count=count, offset=offset)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname, count=-1, offset=0):
    return ivecs_read(fname, count=count, offset=offset).view('float32')

def read_seqs(fname):
    seqs = []
    for line in open(fname):
        seqs.append(line.strip())
    return seqs 

class KNN_Dstore(object):
    def __init__(self, args):
        self.k = args.k
        self.index = self.setup_faiss(args)
    
    def setup_faiss(self, args):
        if args.dstore_fvecs is None:
            raise ValueError('Cannot build a datastore without the data.')
        
        start = time.time()
        index_file = args.faiss_index 
        index = faiss.read_index(index_file, faiss.IO_FLAG_ONDISK_SAME_DIR)
        print('Reading datastore took {} s'.format(time.time() - start))
        index.nprobe = args.probe
        
        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if not args.no_load_keys:
            self.keys = fvecs_read(args.dstore_fvecs) 
            # self.vals = np.arange(self.keys.shape[0]) #may take extremely long time to do this
        
        self.vals = read_seqs(args.dstore_seqs)
        
        if args.load_labels:
            self.labels = read_seqs(args.dstore_labels)
        
        return index
    
    def get_retrieved_seqs(self, index): # re_indexes [b, k]
        return self.vals[index]
    
    def get_retrieved_labels(self, index):
        return self.labels[index]

    def get_knns(self, queries):
        start = time.time()
        dists, knns = self.index.search(queries.detach().cpu().float().numpy(), self.k)
        return dists, knns
    
    def get_knn_prob(self, queries, dists=None, knns=None ): # queries [batch_size, dimensions]
        if dists is None:
            dists, knns = self.get_knns(queries)
        
        dists = -1 * dists
        dists = torch.from_numpy(dists).to(queries.device)
        probs = F.softmax(dists, dim=-1)
        return probs
