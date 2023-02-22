import esm
import torch
from transformers import BertTokenizer, ESMTokenizer
import sys
sys.path.insert(0, '..')
from protretrieval.retriever import Retriever
from benchmark.utils import get_esm_feature

class retrieve_by_query:
    def __init__(self, retrieval_args, tokenizer_type, feature_type='esm', preprocess_device=0):
        
        tokenizer_dict = {
            'bert': BertTokenizer,
            'esm': ESMTokenizer,
        }
        pretrained_tokenizer_path = {
            'bert':"Rostlab/prot_bert",
            'esm':"facebook/esm-1b"
        }
        tokenizer =  tokenizer_dict[tokenizer_type].from_pretrained(
            pretrained_tokenizer_path[tokenizer_type],
            do_lower_case=False
        )
        retriever = Retriever(retrieval_args, tokenizer)
        
        if feature_type=='esm':
            esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        elif feature_type=='esm_2':
            esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        
        self.retriever = retriever
        self.esm_model = esm_model
        self.alphabet = alphabet
        self.preprocess_device = preprocess_device if torch.cuda.is_available() else 'cpu'

    def retrieve(self, query, need_label=True):
        query_feature = get_esm_feature(query, self.esm_model, self.alphabet, self.preprocess_device)
        dists, re_indexes = self.retriever.knn_dstore.get_knns(query_feature.unsqueeze(0))
        
        retrieve_seqs = []
        retrieve_labels = []

        for index in re_indexes[0]:
            retrieve_seq = self.retriever.knn_dstore.get_retrieved_seqs(index)  
            retrieve_seqs.append(retrieve_seq)
            if need_label:
                retrieve_label = self.retriever.knn_dstore.get_retrieved_labels(index)  
                retrieve_labels.append(retrieve_label)
                
        return retrieve_seqs, retrieve_labels
    
    
def main():
    from analyze_args import parse_args
    args=['--dstore_fvecs', '../retrieval-db/pfam.fvecs', '--faiss_index', '../retrieval-db/pfam.index', '--dstore_seqs',"../retrieval-db/pfam_seq.txt", "--dstore_labels", "../retrieval-db/pfam_label.txt"]
    retrieval_args = parse_args(args)
    retriever = retrieve_by_query(retrieval_args, 'bert')
    query_seq = 'VEWTQQERSIIAGIFANLNYEDIGPKALARCLIVYPWTQRYFGAYGDLSTPDAIKGNAKIAAHGVKVLHGLDRAVKNMDNINEAYSELSVLHSDKLHVDPDNFRILGDCLTVVIAANLGDAFTVETQCAFQKFLAVVVFALGRKYH'
    seqs = retriever.retrieve(query_seq, need_label=True)

    
    
#main()