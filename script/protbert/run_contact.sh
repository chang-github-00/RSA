cd ../../
export WANDB_API_KEY='write your api key here'
export WANDB_PROJECT=protein

MODEL='Rostlab/prot_bert_bfd'
TASK_NAME=contact
MODEL_TYPE=bert
DATA_DIR=tape/data
PT=protbert
BS=1
GS=1
LR=3e-5
WR=0.08
EPOCHS=15
EVAL_STEP=500
SEED=111
MAX_LEN=400
SAVE_STEP=500
OUTPUT_DIR=checkpoints/$MODEL_TYPE-$PT-$TASK_NAME-$LR-$MAX_LEN

python run_pretrained_downstream.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
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
    --fp16 True\
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --report_to wandb\
    --resume_from_checkpoint checkpoints/bert-protbert-contact-3e-5-600 \




