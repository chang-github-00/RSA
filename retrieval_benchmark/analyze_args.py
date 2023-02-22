from transformers  import HfArgumentParser
from dataclasses import dataclass, field

@dataclass
class RetrievalTrainingArguments:
    k: int = field(
        default=100,
        metadata={"help": "number of retrieved document for training."}
    )
    
    dstore_fvecs: str = field(
        default=None,
        metadata={"help": "path to store features"}
    )
    faiss_index: str = field(
        default=None,
        metadata={"help": "path to pretrained knn index"}
    )
    probe: int = field(
        default=8,
        metadata={"help": "number of probes for knn."}
    )
    dstore_seqs: str = field(
        default=None,
        metadata={"help": "path to stored sequences"}
    )
    dstore_labels: str = field(
        default=None,
        metadata={"help": "path to stored labels of the sequences"}
    )
    load_labels: bool = field(
        default=True, metadata={"help": "Whether need to load labels (if notation is available)"}
    )
    no_load_keys: bool = field(
        default=True, metadata={"help": "Whether need to load keys (very very big)"}
    )
    concat_max_len: int = field(
        default=600,
        metadata={"help": "length of protein after concatenation"}
    )

def parse_args(args=None):
    parser = HfArgumentParser((RetrievalTrainingArguments))
    retrieval_args, = parser.parse_args_into_dataclasses(args)
    return retrieval_args   


