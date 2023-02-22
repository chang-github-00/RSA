cd ../../
export WANDB_API_KEY='write your api key here'
export WANDB_PROJECT=protein

MODEL='Rostlab/prot_bert_bfd'
TASK_NAME=contact
MODEL_TYPE=augment_bert
OUTPUT_DIR=checkpoints/$MODEL-$TASK_NAME-$SEED-$OI
DATA_DIR=tape/msa_dataset
BS=1
GS=1
LR=3e-5
WR=0.08
EPOCHS=100
EVAL_STEP=500
SEED=111
MAX_LEN=350
KNN=16
CONCAT_MAX_LEN=500

python run_pretrained_downstream.py \
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
    --seed $SEED \
    --max_len $MAX_LEN \
    --optimizer AdamW \
    --frozen_bert False \
    --fp16 True\
    --k $KNN\
    --concat_max_len $MAX_LEN \
