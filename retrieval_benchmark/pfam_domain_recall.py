import json
import pandas as pd
import numpy as np
from analyze_args import parse_args
from query_retrieval import retrieve_by_query
from tqdm import tqdm
import logging

logging.basicConfig(filename='logger.log', level=logging.INFO)

def retrieval_domain_result(query_json_file, retriever, output_file):
    all_precision = []
    all_recall = []
    
    output_f = open(output_file, 'w')
    
    with open(query_json_file, 'r') as f:
        for line in tqdm(f):
            sample = json.loads(line)
            protein = sample['seq']
            target = sample['domain']
            retrieved_seqs, retrieved_labels = retriever.retrieve(protein, need_label=True)
            retrieved_labels = [label.split()[-1][:7] for label in retrieved_labels]
            recall_labels = np.array([label==target for label in retrieved_labels])
            
            if recall_labels.sum() > 0: 
                all_recall.append(1) 
            else:
                all_recall.append(0) 
            all_precision.append(recall_labels.mean())
            
            write_json = {'target domain' : target, 'retrieved domain': ','.join(retrieved_labels)}
            output_f.write(json.dumps(write_json) + '\n')
            
    recall_rate = np.array(all_recall).mean()
    precision_rate = np.array(all_precision).mean()
    
    
    return recall_rate, precision_rate
            
def main():
    retrieval_args = parse_args()
    logging.info("start building retriever")
    retrieve_query = retrieve_by_query(retrieval_args, 'bert')
    logging.info("finished building retriever")
    recall_rate, precision_rate = retrieval_domain_result('pfam_ir.json', retrieve_query, output_file='pfam_domain_ir_k_100.json')
    
    logging.info("precision rate: {pre}, recall rate: {rec}".format(pre=str(precision_rate), rec=str(recall_rate)))

if __name__ == '__main__':
    main()

    