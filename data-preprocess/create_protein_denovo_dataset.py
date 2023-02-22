from urllib.request import urlretrieve
from tqdm import tqdm
import os
import numpy as np


import lmdb
import pickle as pkl
import warnings
warnings.simplefilter('ignore')
def storeLMDB(items: list, store_path: str):
    print("Store LMDB Format....")
    env = lmdb.open(store_path, map_size=int(1e10), max_readers=1)
    with env.begin(write=True) as txn:
        num_examples = len(items)
        txn.put(b'num_examples', pkl.dumps(num_examples))
        for index in range(num_examples):
            try:
                txn.put(str(index).encode(), pkl.dumps(items[index]))
            except:
                print('error ',index)
    
def read_lmdb(data_file):
    env_in = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False) #SOS

    txn_in = env_in.begin(write=False)
    
    num_examples = pkl.loads(txn_in.get(b'num_examples'))  #SOS
    print(num_examples)
    for index in range(num_examples):
        try:
            item = pkl.loads(txn_in.get(str(index).encode()))
        except:
            pass
    print('finish loading samples')

def download_and_preprocess(pdb_id, chain_name, store_path):
    url = 'https://files.rcsb.org/download/'+pdb_id.upper()+'.pdb'
    try:
        urlretrieve(url, os.path.join('../tape/denovo/raw_dataset',pdb_id+'.pdb'))
    except:
        print("An exception occurred")
        return False

    from Bio.PDB.PDBParser import PDBParser
    p = PDBParser(PERMISSIVE=1)
    s = p.get_structure(pdb_id, os.path.join("../tape/denovo/raw_dataset",pdb_id+'.pdb'))  

    chain = None
    for i in s.get_chains():
        if i.id == chain_name:
            chain = i
    
    if chain is not None:
        from Bio.PDB.PDBIO import PDBIO
        io=PDBIO()
        io.set_structure(chain)
        io.save(os.path.join(store_path, pdb_id+'_'+chain_name+'.pdb'))
        
        residues = chain.get_residues()
        tertiary = []
        for residue in residues:
            tertiary.append(residue.center_of_mass())
        tertiary = np.array(tertiary)
        
        from Bio import SeqIO
        for record in SeqIO.parse(os.path.join(store_path, pdb_id+'_'+chain_name+'.pdb'), "pdb-atom"):
            seq = str(record.seq)
        
        if len(seq) == len(tertiary):
            return seq, tertiary
        else:
            return seq, tertiary[:len(seq)]
    else:
        return None, None


if __name__ == '__main__':
    path = '../tape/'
    de_novo_name_file = path + 'denovo/denovo_target_chains.txt'

    with open(de_novo_name_file) as f:
        de_novo_names = f.readlines()
    de_novo_names = [x.strip() for x in de_novo_names]
    
    cache = dict()

    for i, name in tqdm(enumerate(de_novo_names)):
        print("downloading ", name)
        pdb_id = name.split('_')[0]
        chain_name = name.split('_')[1]
        seq, tertiary = download_and_preprocess(pdb_id, chain_name, path+'denovo/structure/')
        if seq is not None and len(seq)>6:
            item = dict()
            item['primary'] = seq
            item['tertiary'] = tertiary
            item['protein_length'] = len(item['primary'])
            item['valid_mask'] = [True]*len(item['primary'])
            item['id'] = name
            cache[i] = item
        else:
            print("not available")
    
    storeLMDB(cache, '../tape/data/proteinnet/proteinnet_de_novo.lmdb')
    data = read_lmdb('../tape/data/proteinnet/proteinnet_de_novo.lmdb')
    