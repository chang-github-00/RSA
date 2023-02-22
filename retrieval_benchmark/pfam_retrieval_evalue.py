import json
import pandas as pd
import numpy as np
from analyze_args import parse_args
from query_retrieval import retrieve_by_query
from tqdm import tqdm
import logging
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import subprocess

logging.basicConfig(filename='logger.log', level=logging.INFO)

def retrieval_domain_result(query_json_file, retriever, query_dir, output_dir, align_dir):
    id = 0
    with open(query_json_file, 'r') as f:
        line_list =[ line for line in f]
        for line in tqdm(line_list):
            sample = json.loads(line)
            protein = sample['seq']
            target = sample['domain']
            retrieved_seqs, retrieved_labels = retriever.retrieve(protein, need_label=True)
            retrieved_labels = [label.split()[-1][:7] for label in retrieved_labels]
            recall_labels = np.array([label==target for label in retrieved_labels])

            retrieved_records = (SeqRecord(Seq(seq), str(index)) for index,seq in enumerate(retrieved_seqs) )
            SeqIO.write(retrieved_records, output_dir+'/'+str(id)+'.fasta', "fasta")
            SeqIO.write(SeqRecord(Seq(protein), str(0)), query_dir+'/'+str(id)+'.fasta', "fasta")

            out_bytes = subprocess.check_output(["./jackhmmer", query_dir+'/'+str(id)+'.fasta', output_dir+'/'+str(id)+'.fasta'])
            out_text = out_bytes.decode('utf-8')
            
            output_f = open(align_dir+'/'+str(id)+'.output','w')
            output_f.write(out_text)

            id += 1
            
def main():
    retrieval_args = parse_args()
    logging.info("start building retriever")
    retrieve_query = retrieve_by_query(retrieval_args, 'bert')
    logging.info("finished building retriever")
    retrieval_domain_result('pfam_ir.json', retrieve_query, output_dir='retrieved_seqs', query_dir='query_seqs', align_dir='retrieved_alignments')

if __name__ == '__main__':
    main()

    