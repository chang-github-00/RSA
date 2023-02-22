cd ../../
export WANDB_API_KEY='write your api key here'
export WANDB_PROJECT=protein

TODO=test
MODEL='Rostlab/prot_bert_bfd'
TASK_NAME=remote_homology
MODEL_TYPE=augment_bert
DATA_DIR=tape/msa_dataset_pfam
PT=protbert
BS=1
GS=1
LR=3e-6
WR=0
EPOCHS=100
EVAL_STEP=500
SAVE_STEP=500
SEED=111
MAX_LEN=600
CONCAT_MAX_LEN=800
SELECT_METHOD=first
NUM_MSA=16
OUTPUT_DIR=checkpoints/$MODEL_TYPE-$PT-$TASK_NAME-$LR-$WR-$MAX_LEN-$NUM_MSA-$SELECT_METHOD
LOAD_CKPT=checkpoints/''

if [[ $TODO == test ]]
then
    echo "Testing..."
    python run_pretrained_downstream.py \
            --model_type $MODEL_TYPE \
            --tokenizer_name $MODEL \
            --model_name_or_path $MODEL \
            --task_name $TASK_NAME \
            --output_dir $OUTPUT_DIR \
            --data_dir $DATA_DIR \
            --do_train True  \
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
            --fp16 True\
            --select_method $SELECT_METHOD \
            --num_msa $NUM_MSA \
            --concat_max_len $CONCAT_MAX_LEN \
            --save_total_limit 2 \
            --report_to wandb \
            --resume_from_checkpoint $LOAD_CKPT
else
    echo "Training..."
    python run_pretrained_downstream.py \
            --model_type $MODEL_TYPE \
            --tokenizer_name $MODEL \
            --model_name_or_path $MODEL \
            --task_name $TASK_NAME \
            --output_dir $OUTPUT_DIR \
            --data_dir $DATA_DIR \
            --do_train True  \
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
            --select_method $SELECT_METHOD \
            --num_msa $NUM_MSA \
            --concat_max_len $CONCAT_MAX_LEN \
            --save_total_limit 2 \
            --report_to wandb 
fi
