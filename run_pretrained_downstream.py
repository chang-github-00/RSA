# import faiss
from typing import Optional

from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments, BertTokenizer, ESMTokenizer, set_seed, Trainer
import logging
import transformers
from benchmark.pretrained_models import model_mapping, load_adam_optimizer_and_scheduler
from benchmark.dataset import dataset_mapping, output_modes_mapping
from benchmark.config import config_mapping
from benchmark.metrics import build_compute_metrics_fn
from benchmark.trainer import OntoProteinTrainer
from protretrieval.trainer import KNNContactTrainer, KNNTrainer, KNNInteractionTrainer
from protretrieval.retrieval_model_wrapper import KNNProteinModel, KNNProteinModelParallel
from msa_augment import AugmentContactTrainer, AugmentTrainer
from msa_augment import MSAAugmentedProteinModel
import pickle as pkl
import warnings
import os
warnings.filterwarnings("ignore")
#os.environ["WANDB_DISABLED"] = "true"
import wandb
from msa import MSATokenizer
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logger = logging.getLogger(__name__)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# DEVICE = "cuda"
transformers.logging.set_verbosity_error()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_type: str=field(
        default=None,
        metadata={"help": "model type: esm/bert"}
    )
    
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    mean_output: bool = field(
        default=True, metadata={"help": "output of bert, use mean output or pool output"}
    )

    optimizer: str = field(
        default="AdamW",
        metadata={"help": "use optimizer: AdamW(True) or Adam(False)."}
    )

    frozen_bert: bool = field(
        default=False,
        metadata={"help": "frozen bert model."}
    )
    
    hidden_size: int = field(
        default=768,
        metadata={"help": "hidden size of transformer model."}
    )
    
    num_hidden_layers: int = field(
        default=12,
        metadata={"help": "num hidden layers of transformer model."}
    )
    
    num_attention_heads: int = field(
        default=12,
        metadata={"help": "num attention heads of transformer model."}
    )
    
    num_attention_heads: int = field(
        default=12,
        metadata={"help": "num attention heads of transformer model."}
    )
    
    intermediate_size: int = field(
        default=3072,
        metadata={"help": "intermediate size of transformer model."}
    )
    
    vocab_size: int = field(
        default=30,
        metadata={"help": "vocab size of transformer model."}
    )
    


@dataclass
class DynamicTrainingArguments(TrainingArguments):
    # For ensemble
    experiment_name: str = field(
        default='training_job',
        metadata={"help": "experiment name for writing into wandb."}
    )

    save_strategy: str = field(
        default='steps',
        metadata={"help": "The checkpoint save strategy to adopt during training."}
    )

    save_steps: int = field(
        default=500,
        metadata={"help": " Number of updates steps before two checkpoint saves"}
    )

    evaluation_strategy: str = field(
        default='steps',
        metadata={"help": "The evaluation strategy to adopt during training."}
    )

    eval_steps: int = field(
        default=100,
        metadata={"help": "Number of update steps between two evaluations"}
    )

    save_logit: bool = field(
        default=False,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )

    save_logit_dir: str = field(
        default=None,
        metadata={"help": "Where to save the prediction result"}
    )

    # Regularization
    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )

    evaluate_during_training: bool = field(
        default=True,
        metadata={"help": "evaluate during training."}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "If a value is passed, will limit the total amount of checkpoints."}
    )

    resume_from_checkpoint: str=field(default='')

    fp16 = True


@dataclass
class BTDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(dataset_mapping.keys())})
    data_dir: str = field(
        default=None,
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_len: int = field(
        default=1e10, metadata={"help": "max lengths of protein sequences"}
    )
    need_features: bool = field(
        default=False,
        metadata={"help": "whether need to extract esm features of datasets."}
    )
    feature_type: str = field(
        default='esm',
        metadata={"help": "feature type for retrieval"}
    )
    preprocess_device: int = field(
        default=0,
        metadata={"help": "which gpu to use during preprocessing."}
    )

    select_method: str = field(default='random')

    num_msa: int= field(default=10)



    def __post_init__(self):
        self.task_name = self.task_name.lower()

@dataclass
class RetrievalTrainingArguments:
    k: int = field(
        default=32,
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
        default=False, metadata={"help": "Whether need to load labels (if notation is available)"}
    )
    no_load_keys: bool = field(
        default=False, metadata={"help": "Whether need to load keys (very very big)"}
    )
    concat_max_len: int = field(
        default=600,
        metadata={"help": "length of protein after concatenation"}
    )
    weight: str = field(
        default='distance',
        metadata={"help": "weighting method for retrieval loss"}
    )
    parallel: bool = field(
        default=False,
        metadata={"help": "whether to use parallel retrieval."}
    )

def main():
    parser = HfArgumentParser((ModelArguments, BTDataTrainingArguments, DynamicTrainingArguments, RetrievalTrainingArguments))
    model_args, data_args, training_args, retrieval_args = parser.parse_args_into_dataclasses()
    if training_args.report_to == 'wandb':
        wandb.init(project='protein_retrieval', name=training_args.experiment_name)

    retrieval=False
    msa=False
    augment=False
    
    if 'retrieval' in model_args.model_type:
        retrieval = True
        data_args.need_features = True
        model_args.model_type = model_args.model_type.split('_')[-1] # retrieval_esm / retrieval_protbert

    elif 'msa' in model_args.model_type:
        msa = True
        
    elif 'augment' in model_args.model_type:
        augment = True
        data_args.concat_max_len = retrieval_args.concat_max_len 
        #data_args.num_msa = retrieval_args.k
        model_args.model_type = model_args.model_type.split('_')[-1] # augment_esm / augment_protbert

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Check save path
    # if (
    #         os.path.exists(training_args.output_dir)
    #         and os.listdir(training_args.output_dir)
    #         and training_args.do_train
    #         and not training_args.overwrite_output_dir
    # ):
    #     raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s",
        training_args.local_rank,
        # DEVICE,
        training_args.n_gpu,
        bool(training_args.local_rank != -1)
    )

    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    try:
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("Task name: {}, output mode: {}".format(data_args.task_name, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load dataset
    tokenizer_type = {
        'bert': BertTokenizer,
        'esm': ESMTokenizer,
        'msa':MSATokenizer,
    }
    tokenizer = tokenizer_type[model_args.model_type].from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        do_lower_case=False
    )
    
    if not augment:
        processor = dataset_mapping[model_args.model_type][data_args.task_name](max_len=data_args.max_len, tokenizer=tokenizer,
                                                        need_features=data_args.need_features, preprocess_device = data_args.preprocess_device,
                                                                                select_method=data_args.select_method,num_msa=data_args.num_msa)
    else:
        processor = dataset_mapping['augment'][data_args.task_name](max_len=data_args.max_len, concat_max_len=data_args.concat_max_len, tokenizer=tokenizer,
                                                        need_features=data_args.need_features, preprocess_device = data_args.preprocess_device,
                                                                                select_method=data_args.select_method,num_msa=data_args.num_msa)
        # For classification task, num labels is determined by specific tasks
    # For regression task, num labels is 1.
    num_labels = len(processor.get_labels())
    train_dataset = (
        processor.get_train_examples(data_dir=data_args.data_dir)
    )
    eval_dataset = (
        processor.get_dev_examples(data_dir=data_args.data_dir)
    )
    # eval_dataset=[eval_dataset[i] for i in range(10)]
    if data_args.task_name == 'remote_homology':
        test_fold_dataset = (
            processor.get_test_examples(data_dir=data_args.data_dir, data_cat='test_fold_holdout')
        )
        test_family_dataset = (
            processor.get_test_examples(data_dir=data_args.data_dir, data_cat='test_family_holdout')
        )
        test_superfamily_dataset = (
            processor.get_test_examples(data_dir=data_args.data_dir, data_cat='test_superfamily_holdout')
        )
    elif data_args.task_name == 'contact':
        test_dataset = (
            processor.get_test_examples(data_dir=data_args.data_dir, data_cat='test')
        )
        try:
            denovo_dataset = (
                processor.get_test_examples(data_dir=data_args.data_dir, data_cat='de_novo')
            )
        except:
            print("error finding denovo dataset")
    elif data_args.task_name == 'ss3' or data_args.task_name == 'ss8':
        print(data_args.task_name + ' test_dataset')
        try:
            cb513_dataset = (
                processor.get_test_examples(data_dir=data_args.data_dir, data_cat='cb513')
            )
            # ts115_dataset = (
            #     processor.get_test_examples(data_dir=data_args.data_dir, data_cat='ts115')
            # )
            casp12_dataset = (
                processor.get_test_examples(data_dir=data_args.data_dir, data_cat='casp12')
            )
            denovo_dataset = (
                processor.get_test_examples(data_dir=data_args.data_dir, data_cat='de_novo')
            )
        except:
            print('missing test set of ss')
    else:
        test_dataset = (
            processor.get_test_examples(data_dir=data_args.data_dir, data_cat='test')
        )
    #print(model_args.model_type)
    #print(data_args.task_name)
    model_fn = model_mapping[model_args.model_type][data_args.task_name]
    
    if model_args.model_name_or_path is not None:
        model = model_fn.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            mean_output=model_args.mean_output
        )

    if model_args.frozen_bert:
        #unfreeze_layers = []
        unfreeze_layers = ['layer.29', 'bert.pooler', 'classifier']
        for name, parameters in model.named_parameters():
            parameters.requires_grad = False
            for tags in unfreeze_layers:
                if tags in name:
                    parameters.requires_grad = True
                    break

    if retrieval:
        if retrieval_args.parallel and data_args.task_name in ['contact', 'stability', 'fluorescence', 'remote_homology','ppi']:
            model = KNNProteinModelParallel(data_args, model_args, retrieval_args, model, tokenizer)
        else: 
            model = KNNProteinModel(data_args, model_args, retrieval_args, model, tokenizer)
        
    if augment:
        model = MSAAugmentedProteinModel(data_args, model_args, retrieval_args, model, tokenizer)
        
    if data_args.task_name == 'stability' or data_args.task_name == 'fluorescence':
        training_args.metric_for_best_model = "eval_spearmanr"
    elif data_args.task_name == 'remote_homology':
        training_args.metric_for_best_model = "eval_accuracy"
    else:
        pass

    
    if not retrieval and not augment:
        if data_args.task_name == 'contact':
            # training_args.do_predict=False
            trainer = OntoProteinTrainer(
                # model_init=init_model,
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=build_compute_metrics_fn(data_args.task_name, output_type=output_mode),
                data_collator=train_dataset.collate_fn,
                optimizers=load_adam_optimizer_and_scheduler(model, training_args, train_dataset) if model_args.optimizer=='Adam' else (None, None)
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=build_compute_metrics_fn(data_args.task_name, output_type=output_mode),
                data_collator=train_dataset.collate_fn,
                optimizers=load_adam_optimizer_and_scheduler(model, training_args, train_dataset) if model_args.optimizer=='Adam' else (None, None)
            )
    elif augment:
        if data_args.task_name == 'contact':
            trainer = AugmentContactTrainer(
                # model_init=init_model,
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=build_compute_metrics_fn(data_args.task_name, output_type=output_mode),
                data_collator=train_dataset.collate_fn,
                optimizers=load_adam_optimizer_and_scheduler(model, training_args, train_dataset) if model_args.optimizer=='Adam' else (None, None)
            )
        else:
            trainer = AugmentTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=build_compute_metrics_fn(data_args.task_name, output_type=output_mode),
                data_collator=train_dataset.collate_fn,
                optimizers=load_adam_optimizer_and_scheduler(model, training_args, train_dataset) if model_args.optimizer=='Adam' else (None, None)
            )
    else:
        if data_args.task_name == 'contact':
            trainer = KNNContactTrainer(
                # model_init=init_model,
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=build_compute_metrics_fn(data_args.task_name, output_type=output_mode),
                data_collator=train_dataset.collate_fn,
                optimizers=load_adam_optimizer_and_scheduler(model, training_args, train_dataset) if model_args.optimizer=='Adam' else (None, None)
            )
        elif data_args.task_name == 'ppi':
            trainer = KNNInteractionTrainer(
                # model_init=init_model,
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=build_compute_metrics_fn(data_args.task_name, output_type=output_mode),
                data_collator=train_dataset.collate_fn,
                optimizers=load_adam_optimizer_and_scheduler(model, training_args, train_dataset) if model_args.optimizer=='Adam' else (None, None)
            )
            
        else:
            trainer = KNNTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=build_compute_metrics_fn(data_args.task_name, output_type=output_mode),
                data_collator=train_dataset.collate_fn,
                optimizers=load_adam_optimizer_and_scheduler(model, training_args, train_dataset) if model_args.optimizer=='Adam' else (None, None)
            )
    # Training
    if training_args.do_train:
        # pass
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint,ignore_keys_for_eval=['attentions','hidden_states'])
        trainer.save_model(training_args.output_dir)
        try:
            tokenizer.save_pretrained(training_args.output_dir)
        except:
            print("Error when saving tokenizer")

    # Prediction
    logger.info("**** Test ****")

    # trainer.compute_metrics = metrics_mapping(data_args.task_name)
    if data_args.task_name == 'remote_homology':
        predictions_fold_family, input_ids_fold_family, metrics_fold_family = trainer.predict(test_fold_dataset)
        predictions_family_family, input_ids_family_family, metrics_family_family = trainer.predict(test_family_dataset)
        predictions_superfamily_family, input_ids_superfamily_family, metrics_superfamily_family = trainer.predict(test_superfamily_dataset)
        print("metrics_fold: ", metrics_fold_family)
        print("metrics_family: ", metrics_family_family)
        print("metrics_superfamily: ", metrics_superfamily_family)
    elif data_args.task_name == 'contact':

        predictions_denovo, input_ids_denovo, metrics_denovo = trainer.predict(denovo_dataset)
        print("metrics denovo", metrics_denovo)
        if predictions_denovo is not None:
            print(predictions_denovo)
        pkl.dump(predictions_denovo, open(os.path.join(training_args.output_dir, "predictions_denovo.pkl"), "wb"))
        # except:
        #     print("dont support de novo")
        predictions_family, input_ids_family, metrics_family = trainer.predict(test_dataset)
        print("metrics", metrics_family)
    elif data_args.task_name == 'ss8' or data_args.task_name == 'ss3':
        predictions_cb513, input_ids_cb513, metrics_cb513 = trainer.predict(cb513_dataset)
        #predictions_ts115, input_ids_ts115, metrics_ts115 = trainer.predict(ts115_dataset)
        predictions_casp12, input_ids_casp12, metrics_casp12 = trainer.predict(casp12_dataset)
        predictions_denovo=None
        try:
            predictions_denovo, input_ids_denovo, metrics_denovo = trainer.predict(denovo_dataset)
        except:
            print("dont support de novo")
        if predictions_denovo is not None:
            pkl.dump(predictions_denovo, open(os.path.join(training_args.resume_from_checkpoint, "predictions_denovo_ss8.pkl"), "wb"))
        else:
            print("no output for denovo contact is reported")
        print("cb513: ", metrics_cb513)
        #print("ts115: ", metrics_ts115)
        print("casp12: ", metrics_casp12)
    else:
        predictions_family, input_ids_family, metrics_family = trainer.predict(test_dataset)
        print("metrics", metrics_family)


if __name__ == '__main__':
    main()
