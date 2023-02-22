from tape.datasets import pad_sequences, dataset_factory
import re
from scipy.spatial.distance import squareform, pdist

import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn.functional as F
import torch.nn.modules as modules
from torch.autograd import Variable
from torch import optim
import random
    

def parse_fasta(filename):
  '''function to parse fasta file'''
  header = []
  sequence = []
  lines = open(filename, "r")
  for line in lines:
    line = line.rstrip()
    if line[0] == ">":
      header.append(line[1:])
      sequence.append([])
    else:
      sequence[-1].append(line)
  lines.close()
  sequence = [''.join(seq) for seq in sequence]
  return np.array(header), np.array(sequence)

def filt_gaps(msa, states, gap_cutoff=0.5):
  '''filters alignment to remove gappy positions'''
  frac_gaps = np.mean((msa == states-1).astype(np.float),0)
  non_gaps = np.where(frac_gaps < gap_cutoff)[0]
  return msa[:,non_gaps], non_gaps

def get_eff(msa, eff_cutoff=0.8):
  '''compute effective weight for each sequence'''  
  msa_sm = 1.0 - squareform(pdist(msa,"hamming"))
  msa_w = (msa_sm >= eff_cutoff).astype(np.float)
  msa_w = 1.0/np.sum(msa_w,-1)
  return msa_w

def one_hot(msa,states):
  one = np.eye(states)
  return one[msa]

def mk_msa(seqs, max_num=16):
  '''one hot encode msa'''
  alphabet = "ARNDCQEGHILKMFPSTWYV-" 
  states = len(alphabet)
  a2n = {}
  for a,n in zip(alphabet,range(states)):
    a2n[a] = n

  def aa2num(aa):
    '''convert aa into num'''
    if aa in a2n: return a2n[aa]
    else: return a2n['-']
  ################
  
  msa = []
  length = len(seqs[0])
  for seq in seqs:
    temp = [aa2num(aa) for aa in seq]
    if len(temp)>=length:
        temp = temp[:length]
    else:
        temp = temp + [a2n['-']]*(length-len(temp))
    msa.append(temp)
  
  msa = msa[:max_num]
  msa_ori = np.array(msa)
  #msa_ori, v_idx = filt_gaps(msa_ori, states)
  return msa_ori, one_hot(msa_ori,states)


def get_mtx(W):
    # l2norm of 20x20 matrices (note: we ignore gaps)
    raw = np.sqrt(np.sum(np.square(W[:,:,:,:]),(1,3)))
    np.fill_diagonal(raw,0)
    # apc (average product correction)
    ap = np.sum(raw,0,keepdims=True)*np.sum(raw,1,keepdims=True)/np.sum(raw)
    apc = raw - ap
    np.fill_diagonal(apc,0)
    
    return(raw,apc)

if __name__ == '__main__':
    data = dataset_factory('tape/data/proteinnet/proteinnet_test.lmdb', 'data')
    precision = []
        
    for i in range(len(data)):
        correct = 0
        total = 0
        item = data[i]
        contact_map = np.less(squareform(pdist(torch.tensor(item['tertiary']))), 8.0).astype(np.int64)
        origin_contact_map = contact_map.copy()
        yind, xind = np.indices(contact_map.shape)
        valid_mask = item['valid_mask']
        invalid_mask = ~(valid_mask[:, None] & valid_mask[None, :])
        invalid_mask |= np.abs(yind - xind) < 6
        contact_map[invalid_mask] = -1
        #print(contact_map)
        
        msa_file = 'tape/msa/proteinnet/proteinnet_test/{}.a3m'.format(i)
        
        names,seqs = parse_fasta(msa_file)
        msa_ori, msa = mk_msa(seqs)
        
        # collecting some information about input msa
        nrow = msa.shape[0] # number of sequences
        ncol = msa.shape[1] # length of sequence
        states = msa.shape[2] # number of states (or categories)
        
        #enviroment setting
        device = torch.device("cpu") # Uncomment this to run on GPU

        MSA_0 = torch.from_numpy(msa.astype(np.float32))
        MSA = torch.reshape(MSA_0,(-1,ncol*states))
        
        W0 = Variable(torch.zeros(ncol*states,ncol*states), requires_grad=True)
        
        MASK = (1.0 - torch.eye(ncol)[:,None,:,None]) * torch.ones((states,states))[None,:,None,:]
        MASK = MASK.reshape((ncol*states,ncol*states))

        b = Variable(torch.zeros(ncol*states), requires_grad=True)

        learning_rate = 5e-4
        

        
        for t in range(100):
  
            W = (W0 + W0.transpose(1,0))/2 * MASK 
            MSA_pred = MSA.mm(W) + b
            
            MSA_pred = torch.reshape(MSA_pred,(-1,ncol,states))
            loss = torch.sum(- MSA_0 * F.log_softmax(MSA_pred, -1))
            reg_b = 0.01 * (b*b).sum()
            reg_w = 0.01 * 0.5 * states * ncol * (W*W).sum()
            loss = loss  + reg_b + reg_w
            loss.backward()
            
            if (t) % (int(100/10)) == 0: 
                print(t, loss.item())
            
            # update the gradient
            with torch.no_grad():
                W0 -= learning_rate * W0.grad
                b -= learning_rate * b.grad
                # Manually zero the gradients after updating weights
                W0.grad.zero_()
        
        w = torch.Tensor.cpu(W).detach().numpy()
        w = np.reshape(w,(ncol,states,ncol,states))
        raw, apc = get_mtx(w)
                
        raw, apc = get_mtx(w)
        
        # positive = (origin_contact_map == 1).sum()
        # negative = (origin_contact_map == 0).sum()
        # per = positive/(positive+negative)

        output = torch.tensor(raw)
        contact_map = torch.tensor(contact_map)
        
        valid_mask = (contact_map != -1)
        output  = output * valid_mask
        most_likely = output.topk(ncol // 5, sorted=False)
        selected = contact_map.gather(0, most_likely.indices)
        correct += (selected>=0).sum().float()
        total += (selected>=0).numel()

        precision.append(float(correct/total))
        print(str(i)+"th protein precision l5 is {}".format(correct/total))
    print(np.array(precision).mean())
    # compute the precision at l/5
        
        
        
        