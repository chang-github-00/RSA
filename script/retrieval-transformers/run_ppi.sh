cd ../../
export WANDB_API_KEY='write your api key here'
export WANDB_PROJECT=protein

TODO=train
MODEL='Rostlab/prot_bert_bfd'
TASK_NAME=ppi
MODEL_TYPE=retrieval_bert
DATA_DIR=tape/data
PT=no-pretrained-small
BS=1
GS=1
LR=1e-5
WR=0
EPOCHS=200
EVAL_STEP=500
SEED=111
MAX_LEN=600
CONCAT_MAX_LEN=800
SAVE_STEP=500
KNN=16
FVECS_DIR=retrieval-db/pfam.fvecs
INDEX_DIR=retrieval-db/pfam.index
SEQ_DIR=retrieval-db/pfam_seq.txt
NPROBE=8
OUTPUT_DIR=checkpoints/$MODEL_TYPE-$PT-$TASK_NAME-$LR-$WR-$MAX_LEN-$CONCAT_MAX_LEN-$KNN
LOAD_CKPT=to-be-added

if [[ $TODO == test ]]
then
    echo "Testing..."
    python run_downstream.py \
    --model_type $MODEL_TYPE \
    --tokenizer_name $MODEL \
    --model_name_or_path $MODEL \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --data_dir $DATA_DIR \
    --do_train True \
    --do_predict True \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS \
    --gradient_accumulation_steps $GS \
    --learning_rate $LR \
    --num_train_epochs 1 \
    --warmup_ratio $WR \
    --logging_steps $EVAL_STEP \
    --eval_steps $EVAL_STEP \
    --save_steps $SAVE_STEP \
    --seed $SEED \
    --max_len $MAX_LEN \
    --optimizer AdamW \
    --frozen_bert False \
    --fp16 False\
    --k $KNN\
    --dstore_fvecs $FVECS_DIR \
    --faiss_index $INDEX_DIR \
    --probe $NPROBE \
    --dstore_seqs $SEQ_DIR \
    --no_load_keys True \
    --concat_max_len $CONCAT_MAX_LEN \
    --preprocess_device 0 \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --report_to wandb \
    --resume_from_checkpoint $LOAD_CKPT
else    
    echo "Training..."
    python run_downstream.py \
        --model_type $MODEL_TYPE \
        --model_name_or_path $MODEL \
        --task_name $TASK_NAME \
        --output_dir $OUTPUT_DIR \
        --data_dir $DATA_DIR \
        --do_train True \
        --do_predict True \
        --per_device_train_batch_size $BS \
        --per_device_eval_batch_size $BS \
        --gradient_accumulation_steps $GS \
        --learning_rate $LR \
        --num_train_epochs $EPOCHS \
        --warmup_ratio $WR \
        --logging_steps $EVAL_STEP \
        --eval_steps $EVAL_STEP \
        --save_steps $SAVE_STEP \
        --seed $SEED \
        --max_len $MAX_LEN \
        --optimizer AdamW \
        --frozen_bert False \
        --fp16 True\
        --k $KNN\
        --dstore_fvecs $FVECS_DIR \
        --faiss_index $INDEX_DIR \
        --probe $NPROBE \
        --dstore_seqs $SEQ_DIR \
        --no_load_keys True \
        --concat_max_len $CONCAT_MAX_LEN \
        --preprocess_device 0 \
        --save_total_limit 2 \
        --load_best_model_at_end True \
        --report_to wandb 
fi