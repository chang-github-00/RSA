import subprocess
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq


from query_retrieval import retrieve_by_query
from analyze_args import RetrievalTrainingArguments
from transformers  import HfArgumentParser
from dataclasses import dataclass, field

def read_fasta(fasta_file): #use SeqIO to read fasta from files
    for record in SeqIO.parse(fasta_file, "fasta"):
        return str(record.seq)
    
import argparse
import os
import subprocess
from tqdm import tqdm
import random


@dataclass
class DataArguments:
    dataset: str = field(
        default=None,
        metadata={"help": "dataset name"}
    )
    
    path: str = field(
        default='./',
        metadata={"help": "working directory"}
    )
    
    type: str = field(
        default='train',
        metadata={"help": "train/valid/test split"}
    )
    
    feature_type: str = field(
        default='esm',
        metadata={"help": "feature type use either esm or esm2"}
    )

def parse_args(args=None):
    parser = HfArgumentParser((RetrievalTrainingArguments, DataArguments))
    retrieval_args, data_args = parser.parse_args_into_dataclasses(args)
    return retrieval_args, data_args  
  
def arg_to_query_dir(path, dataset, type):
    newpath = path + "/query/" +dataset+'/'+ dataset + "_" + type
    print(newpath)
    assert os.path.exists(newpath)
    
    return newpath

def arg_to_output_dir(path, dataset, type):
    newpath = path + "/retrieve/" + dataset+'/'+dataset + "_" + type
    if not os.path.exists(path + "/retrieve/"):
        os.mkdir(path + "/retrieve/")
    if not os.path.exists(path + "/retrieve/"+dataset):
        os.mkdir(path + "/retrieve/"+dataset)   
    return newpath

def arg_to_align_dir(path, dataset, type):
    newpath = path + "/retrieve/align/" + dataset+'/'+dataset + "_" + type
    if not os.path.exists(path + "/retrieve/align/"):
        os.mkdir(path + "/retrieve/align/")
    if not os.path.exists(path + "/retrieve/align/"+dataset):
        os.mkdir(path + "/retrieve/align/"+dataset)   
    return newpath
    
def arg_to_summary_dir(path, dataset, type):
    newpath = path + "/retrieve/summary/" + dataset+'/'+dataset + "_" + type
    if not os.path.exists(path + "/retrieve/summary/"):
        os.mkdir(path + "/retrieve/summary/")
    if not os.path.exists(path + "/retrieve/summary/"+dataset):
        os.mkdir(path + "/retrieve/summary/"+dataset)   
    return newpath   
    
def generate_retrieved_seqs(retriever, path, dataset, type):
    query_dir = arg_to_query_dir(path, dataset, type)
    output_dir = arg_to_output_dir(path, dataset, type)
    align_dir = arg_to_align_dir(path, dataset, type)
    summary_dir = arg_to_summary_dir(path, dataset, type)
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(align_dir):
        os.mkdir(align_dir)
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    for query_file in tqdm(os.listdir(query_dir)):
        if '.seq' not in query_file:
            continue
        file_name = query_file.split('.')[0]
        query = read_fasta(query_dir + "/" + query_file)
        output_file = output_dir + '/' + file_name + '.retrieved'
        aligned_file_temp = align_dir + '/' + file_name + '.aligned'
        aligned_file = align_dir + '/' + file_name + '.a3m'
        summary_file = summary_dir + '/' + file_name + '.summary'
    
        retrieved_seqs, retrieved_labels = retriever.retrieve(query, need_label=True)
        retrieved_records = (SeqRecord(Seq(seq), label) for seq,label in zip(retrieved_seqs, retrieved_labels) )
        SeqIO.write(retrieved_records, output_file, "fasta")

        out_bytes = subprocess.check_output(["./jackhmmer", "-E", "10.0", "-A", aligned_file_temp, query_dir + "/" + query_file, output_file])
        out_text = out_bytes.decode('utf-8')
        
        subprocess.run(['./esl-reformat', '-o', aligned_file, 'a2m', aligned_file_temp])
        
        with open(summary_file,'w') as summary_f:
            summary_f.write(out_text)

        

DATABASE_DIR ={'pfam-esm-1': "../retrieval-db/pfam.index",
               'pfam-esm-2': "../retrieval-db/esm-2/esm_2_pfam.index",
               'scope-esm-1': "../retrieval-db/scope/scope.index"
               }

def main():
    retrieval_args, data_args = parse_args()
    retrieve_query = retrieve_by_query(retrieval_args, 'bert', feature_type=data_args.feature_type)
    generate_retrieved_seqs(retrieve_query, data_args.path, data_args.dataset, data_args.type)

if __name__ == '__main__':
    main()
    
    
    
# run this code to generate the retrieval sequences
# python create_retrieval_seqs.py --k 500 --dataset subcellular_localization --path ../tape --type test --faiss_index ../retrieval-db/pfam.index --dstore_seqs ../retrieval-db/pfam_seq.txt --dstore_labels ../retrieval-db/pfam_label.txt --dstore_fvecs ../retrieval-db/pfam_vectors
# python create_retrieval_seqs.py --k 500 --dataset subcellular_localization --path ../tape --type valid --faiss_index ../retrieval-db/esm-2/esm_2_pfam.index --dstore_seqs ../retrieval-db/pfam_seq.txt --dstore_labels ../retrieval-db/pfam_label.txt --dstore_fvecs ../retrieval-db/esm-2/esm-2_vectors