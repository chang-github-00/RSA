
cd ../../
export WANDB_API_KEY='write your api key here'
export WANDB_PROJECT=protein

TODO=test
MODEL='Rostlab/prot_bert_bfd'
TASK_NAME=stability
MODEL_TYPE=bert
PT=protbert
DATA_DIR=tape/data
BS=1
GS=1
LR=3e-6
WR=0
EPOCHS=40
EVAL_STEP=500
SEED=111
MAX_LEN=60
SAVE_STEP=500
OUTPUT_DIR=checkpoints/$MODEL_TYPE-$PT-$TASK_NAME-$LR-$MAX_LEN
LOAD_CKPT=checkpoints/''

if [[ $TODO == test ]]
then
    echo "Testing..."
    python run_pretrained_downstream.py \
        --model_type $MODEL_TYPE \
        --tokenizer_name $MODEL \
        --model_name_or_path $LOAD_CKPT \
        --task_name $TASK_NAME \
        --output_dir $OUTPUT_DIR \
        --data_dir $DATA_DIR \
        --do_train False \
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
        --fp16 False\
        --save_total_limit 2 \
        --load_best_model_at_end True
else    
    echo "Training..."
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
        --save_steps $SAVE_STEP \
        --seed $SEED \
        --max_len $MAX_LEN \
        --optimizer AdamW \
        --frozen_bert False \
        --fp16 False\
        --save_total_limit 2 \
        --load_best_model_at_end True \
        --report_to wandb
fi