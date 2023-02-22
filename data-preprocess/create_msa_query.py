import lmdb
import pickle as pkl
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
import random
import argparse

# "proteinnet": ["test", "train", "valid"],
# "remote_homology": ["train", "valid", "test_family_holdout", "test_fold_holdout", "test_superfamily_holdout"],
# "stability": ["test", "train", "valid"],
# "secondary_structure": ["casp12", "cb513", "ts115", "train", "valid"],
# "fluorescence": ["test", "train", "valid"]

# for each dataset[type] in db_dic we want to sample 500 seq.

db_dic = {
    "proteinnet": ["test", "train", "valid"],
    "remote_homology": ["train", "valid", "test_family_holdout", "test_fold_holdout", "test_superfamily_holdout"],
    "stability": ["test", "train", "valid"],
    "secondary_structure": ["casp12", "cb513", "ts115", "train", "valid"],
    "fluorescence": ["test", "train", "valid"],
    "subcellular_localization": ["test"],
    "human_ppi": ["test", "train", "valid"],
}

def arg_to_tape_lmdb_file(path, dataset, type):
    newpath = path + "/data/" + dataset + "/"+ dataset + "_" + type + ".lmdb"
    assert os.path.exists(newpath)
    return newpath

def arg_to_output_dir(path, dataset, type):
    newpath = path + "/query/" + dataset + "/"+ dataset + "_" + type 
    if not os.path.exists(path + "/query/"):
        os.mkdir(path + "/query/")
    if not os.path.exists(path + "/query/"+ dataset):
        os.mkdir(path + "/query/"+ dataset)
    if not os.path.exists(newpath):
        os.mkdir(newpath)
    return newpath

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default='./', help="working directory")
    args = parser.parse_args()
    return args

def create_query_dataset(path, dataset, type):
    lmdb_file = arg_to_tape_lmdb_file(path, dataset, type)
    query_dir = arg_to_output_dir(path, dataset, type)
    
    env_in = lmdb.open(str(lmdb_file), max_readers=1, readonly=True,
    lock=False, readahead=False, meminit=False) #SOS
    txn_in = env_in.begin(write=False)

    num_examples = pkl.loads(txn_in.get(b'num_examples'))  #SOS
  
    # index is from 0 to len(sample_lst)-1
    for index in tqdm(range(num_examples)):
        if not os.path.exists(query_dir):
            os.mkdir(query_dir)
        out_path = query_dir + "/" + str(index) + ".seq"
        item = pkl.loads(txn_in.get(str(index).encode()))
        # turn an item (dic) to SeqRecord
        # put relevent info into description
        if dataset=="fluorescence":
            # fluorescence no id 
            protein_length= item['protein_length']
            log_fluorescence = item['log_fluorescence']
            num_mutations = item['num_mutations']
            record = SeqRecord(Seq(item['primary']), str(index), description="fluorescence: " + f'{protein_length},{log_fluorescence},{num_mutations}')
            SeqIO.write(record, out_path, "fasta")
        elif dataset=="proteinnet":
            id = item['id']
            protein_length= item['protein_length']
            record = SeqRecord(Seq(item['primary']), str(id), description="proteinnet: " + f'{protein_length}')
            SeqIO.write(record, out_path, "fasta")
        elif dataset=="stability":
            id = item['id']
            protein_length= item['protein_length']
            topology = item['topology']
            parent = item['parent']
            stability_score = item['stability_score']
            record = SeqRecord(Seq(item['primary']), str(id), description="stability: " + f'{protein_length},{topology},{parent},{stability_score}')
            SeqIO.write(record, out_path, "fasta")
        elif dataset=="remote_homology":
            id = item['id']
            protein_length= item['protein_length']
            class_label= item['class_label']
            fold_label = item['fold_label']
            superfamily_label = item['superfamily_label']
            family_label = item['family_label']
            record = SeqRecord(Seq(item['primary']), str(id), description="remote_homology: " + f'{protein_length},{class_label},{fold_label},{superfamily_label},{family_label}')
            SeqIO.write(record, out_path, "fasta")
        elif dataset=="secondary_structure":
            id = item['id']
            protein_length= item['protein_length']
            record = SeqRecord(Seq(item['primary']), str(id), description="secondary structure: " + f'{protein_length}')
            SeqIO.write(record, out_path, "fasta")
        elif dataset=="subcellular_localization":
            if 'id' not in item:
                id = index
            else:   
                id = item['id']
            protein_length = len(item['primary'])
            record = SeqRecord(Seq(item['primary']), str(id), description="subcellular_localization: " + f'{protein_length}')
            SeqIO.write(record, out_path, "fasta")
        elif dataset=="human_ppi":
            if 'id' not in item:
                id = index
            else:   
                id = item['id']
            protein_length_1 = item['protein_length_1']
            record_1 = SeqRecord(Seq(item['primary_1']), str(id)+"_1", description="human_ppi: " + f'{protein_length_1}')
            protein_length_2 = item['protein_length_2']
            record_2 = SeqRecord(Seq(item['primary_2']), str(id)+"_2", description="human_ppi: " + f'{protein_length_2}')
            SeqIO.write(record_1, query_dir + "/" + str(index)+"_1" + ".seq", "fasta")
            SeqIO.write(record_2, query_dir + "/" + str(index)+"_2" + ".seq", "fasta")
        else:
            print("dataset not supported")
            return
    return num_examples

def main(db_dic):
    args = get_args()
    for ds in list(db_dic.keys()):
        for ty in db_dic[ds]:
            total_num = create_query_dataset(args.path, ds, ty)
            print("-------------------")
            print(f"finish create FULL query data of {ds}_{ty}")
            print(f"Total sample num is {total_num}")
            query_dir = arg_to_output_dir(args.path, ds, ty)
            print(f"query files are saved at {query_dir}")

main(db_dic)

