cd ../../
export WANDB_API_KEY='write your api key here'
export WANDB_PROJECT=protein

TODO=test
MODEL='Rostlab/prot_bert_bfd'
TASK_NAME=stability
MODEL_TYPE=bert
PT=no-pretrained-small
DATA_DIR=tape/data
BS=64
GS=1
LR=1e-5
WR=0.1
EPOCHS=200
EVAL_STEP=500
MAX_LEN=60
SAVE_STEP=500
SEED=111
OUTPUT_DIR=checkpoints/$MODEL_TYPE-$PT-$TASK_NAME-$BS-$LR-$WR-$MAX_LEN
LOAD_CKPT=checkpoints/''


if [[ $TODO == test ]]
then
    echo "Testing..."
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
        --save_total_limit 2 \
        --report_to wandb \
        --resume_from_checkpoint $LOAD_CKPT
else
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
        --save_total_limit 2 \
        # --load_best_model_at_end True \
        --report_to wandb \
        --resume_from_checkpoint $LOAD_CKPT
fi