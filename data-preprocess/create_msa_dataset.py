import lmdb
import pickle as pkl
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import argparse
import json

# db_dic = {
#     "proteinnet": ["test", "train", "valid"],
#     "remote_homology": ["train", "valid", "test_family_holdout", "test_fold_holdout", "test_superfamily_holdout"],
#     "stability": ["test", "train", "valid"],
#     "secondary_structure": ["casp12", "cb513", "ts115", "train", "valid"]
# }

# db_dic = {
#     "proteinnet": ["test", "train", "valid"],
#     "remote_homology": ["train", "valid", "test_family_holdout", "test_fold_holdout", "test_superfamily_holdout"],
#     "stability": ["test", "train", "valid"],
#     "secondary_structure": ["casp12", "cb513", "ts115", "train", "valid"],
#     "fluorescence":["test","train","valid"]
# }

db_dic = {
    # "proteinnet": ["train",],
    #"remote_homology": ["train", "valid", "test_family_holdout", "test_fold_holdout", "test_superfamily_holdout"],
    #"stability": ["test", "train", "valid"],
    #"secondary_structure": [ "cb513",  "train", "valid"],
    # "fluorescence":["test","train","valid"]
    "subcellular_localization": ["test"],
    "human_ppi": ["test"],
}

def convert_lmdb_fasta(data_file, msa_dir, output_file):
    error_id=[]
    env_in = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False) #SOS
    #env_out = lmdb.open(output_file, map_size=int(1e9), max_readers=1)
    
    txn_in = env_in.begin(write=False)
    #txn_out = env_out.begin(write=True)
    
    cache = dict([])
    num_examples = pkl.loads(txn_in.get(b'num_examples'))  #SOS
    #txn_out.put(b'num_examples', pkl.dumps(num_examples))
    
    for index in range(num_examples):
        item = pkl.loads(txn_in.get(str(index).encode()))
        file = msa_dir + '/' + str(index) + '.a3m'
        try:
            msa_seqs = []
            for record in SeqIO.parse(file, "fasta"):
                msa_seqs.append(str(record.seq))
            item["msa"] = msa_seqs
            #txn_out.put(str(index).encode(), pkl.dumps(item))
            cache[index] = item
        except:
            print('error ',index)
            error_id.append(index)
    return cache,error_id

def convert_lmdb_fasta_ppi(data_file, msa_dir, output_file):
    error_id=[]
    env_in = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False) #SOS
    #env_out = lmdb.open(output_file, map_size=int(1e9), max_readers=1)
    
    txn_in = env_in.begin(write=False)
    #txn_out = env_out.begin(write=True)
    
    cache = dict([])
    num_examples = pkl.loads(txn_in.get(b'num_examples'))  #SOS
    #txn_out.put(b'num_examples', pkl.dumps(num_examples))
    
    for index in range(num_examples):
        item = pkl.loads(txn_in.get(str(index).encode()))
        file_1 = msa_dir + '/' + str(index) + '_1.a3m'
        file_2 = msa_dir + '/' + str(index) + '_2.a3m'
        try:
            msa_seqs_1 = []
            for record in SeqIO.parse(file_1, "fasta"):
                msa_seqs_1.append(str(record.seq))
            item["msa_1"] = msa_seqs_1
            
            msa_seqs_2 = []
            for record in SeqIO.parse(file_2, "fasta"):
                msa_seqs_2.append(str(record.seq))
            item["msa_2"] = msa_seqs_2
            #txn_out.put(str(index).encode(), pkl.dumps(item))
            cache[index] = item
        except:
            print('error ',index)
            error_id.append(index)
    return cache,error_id

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
    for index in range(num_examples):
        try:
            item = pkl.loads(txn_in.get(str(index).encode()))
        except:
            pass
    print('finish loading samples')
        

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='./', help="working directory")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    error_key={}
    for ds in list(db_dic.keys()):
        if ds == "human_ppi":
            for ty in db_dic[ds]:
                data_file = args.path + "/data/" + ds + "/"+ ds + "_" + ty + ".lmdb"
                msa_dir = args.path +  "/msa/"  + ds + "/" + ds + "_" + ty
                output_file = args.path + "/msa_dataset/" + ds + "/"+ ds + "_" + ty + ".lmdb"
                
                print(msa_dir)
                assert os.path.exists(data_file)
                assert os.path.exists(msa_dir)
                if not os.path.exists(args.path + "/msa_dataset/"):
                    os.mkdir(args.path + "/msa_dataset/")

                if not os.path.exists(args.path + "/msa_dataset/" + ds):
                    os.mkdir(args.path + "/msa_dataset/" + ds)

                if not os.path.exists(output_file):
                    cache,error_id=convert_lmdb_fasta_ppi(data_file, msa_dir, output_file)
                    storeLMDB(cache, output_file)
                    error_key[ds+'_'+ty]=error_id
                else:
                    print('already exist')
                    read_lmdb(output_file)
        else:
            for ty in db_dic[ds]:
                data_file = args.path + "/data/" + ds + "/"+ ds + "_" + ty + ".lmdb"
                msa_dir = args.path +  "/msa/"  + ds + "/" + ds + "_" + ty
                output_file = args.path + "/msa_dataset/" + ds + "/"+ ds + "_" + ty + ".lmdb"
                
                print(msa_dir)
                assert os.path.exists(data_file)
                assert os.path.exists(msa_dir)
                if not os.path.exists(args.path + "/msa_dataset/"):
                    os.mkdir(args.path + "/msa_dataset/")

                if not os.path.exists(args.path + "/msa_dataset/" + ds):
                    os.mkdir(args.path + "/msa_dataset/" + ds)
                # try:
                cache,error_index = convert_lmdb_fasta(data_file, msa_dir, output_file)
                print('loaded ', msa_dir)
                storeLMDB(cache, output_file)
                print('saved to ',output_file)
                read_lmdb(output_file)
                # except:
                #     print('error')
                error_key[ds+'+'+ty]=error_index
    b = json.dumps(error_key)
    f2 = open('error_key.json', 'w')
    f2.write(b)
    f2.close()