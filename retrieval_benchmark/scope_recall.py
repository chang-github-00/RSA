import json
import pandas as pd
import numpy as np
from analyze_args import parse_args
from query_retrieval import retrieve_by_query
from tqdm import tqdm
import logging

def retrieval_domain_result(query_json_file, retriever, output_file):
    output_f = open(output_file, 'w')
    ntotal = 0
    with open(query_json_file, 'r') as f:
        for line in f:
            ntotal+=1
            
    with open(query_json_file, 'r') as f:
        for line in tqdm(f, total=ntotal):
            sample = json.loads(line)
            protein = sample['seq']
            target = sample['label']
            retrieved_seqs, retrieved_labels = retriever.retrieve(protein, need_label=True)
            #retrieved_labels = [label.split()[-1][:7] for label in retrieved_labels]
            
            write_json = {'target domain' : target, 'retrieved domain': '|'.join(retrieved_labels)}
            output_f.write(json.dumps(write_json) + '\n')

            
def main():
    retrieval_args = parse_args()
    retrieve_query = retrieve_by_query(retrieval_args, 'bert')
    recall_rate, precision_rate = retrieval_domain_result('scope_ir.json', retrieve_query, output_file='scope_ir_k_100.json')
    
    logging.info("precision rate: {pre}, recall rate: {rec}".format(pre=str(precision_rate), rec=str(recall_rate)))

if __name__ == '__main__':
    main()

    