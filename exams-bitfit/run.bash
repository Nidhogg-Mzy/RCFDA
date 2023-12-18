#!/bin/bash

MODEL_TYPE="bert"
TASK_NAME="exams"
TRAIN_OUTPUT_SUBDIR="tensorboard"

RUN_SETTING_NAME="bert_$(date +"%Y%m%d_%H%M%S")"

TRAINED_MODEL_DIR="/data/clhuang/5212/exams-qa/model" # this should be the dir instead of that bin file. They need those config file to run

TRAIN_DATA_DIR="./data/exams/multilingual"

TRAIN_OUTPUT="./outputs"

MAX_SEQ_LENGTH=128

# OUTPUT_DIR="/data/clhuang/5212/exams-qa/outputs"
# CACHE_DIR="/data/clhuang/5212/exams-qa/cache_dir"
# LOG_FILE="/data/clhuang/5212/exams-qa/terminal_outputs/script_log_$(date +"%Y%m%d_%H%M%S").txt"

OUTPUT_DIR="./outputs"
CACHE_DIR="./cache_dir"
LOG_FILE="./terminal_outputs/script_log_$(date +"%Y%m%d_%H%M%S").txt"


python ./scripts/experiments/run_multiple_choice.py \
    --model_type $MODEL_TYPE \
    --task_name $TASK_NAME \
    --tb_log_dir runs/${TRAIN_OUTPUT_SUBDIR}/$RUN_SETTING_NAME \
    --model_name_or_path $TRAINED_MODEL_DIR \
    --do_train \
    --max_seq_length $MAX_SEQ_LENGTH \
    --do_eval \
    --evaluate_during_training \
    --data_dir $TRAIN_DATA_DIR \
    --max_seq_length $MAX_SEQ_LENGTH \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    # >$LOG_FILE 2>&1