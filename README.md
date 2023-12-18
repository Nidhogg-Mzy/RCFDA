# RCFDA
Reducing Catastrophic Forgetting for Domain Adaptation in Multi-choice Q&amp;A

## Dataset

The exams-qa dataset can be found [here](https://github.com/mhardalov/exams-qa).
The arXiv-10 dataset can be found [here](https://paperswithcode.com/dataset/arxiv-10).



## Environment setup

``````shell
conda install transformers pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install adapters tqdm
``````
## Train
``````shell
python ./scripts/experiments/run_multiple_choice.py \
    --model_type $MODEL_TYPE \
    --task_name $TASK_NAME \
    --tb_log_dir runs/${TRAIN_OUTPUT_SUBDIR}/$RUN_SETTING_NAME \
    --model_name_or_path $TRAINED_MODEL_DIR \
    --do_train \
    --do_eval \
    --warmup_proportion ${WARM_UP} \
    --evaluate_during_training \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${LOGGING_STEPS} \
    --data_dir $TRAIN_DATA_DIR \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $MAX_EPOCHS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --output_dir $TRAIN_OUTPUT \
    --weight_decay $WEIGHT_DECAY \
    --overwrite_cache \
    --per_gpu_eval_batch_size=$EVAL_BATCH_SIZE \
    --per_gpu_train_batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --overwrite_output 
``````

## Evaluation

``````python
python evaluate_exams.py \
    --predictions_path predictions.json \
    --dataset_path dev.jsonl \
    --granularity all \
    --output_path results.json
``````

## Prediction

``````python
python ./scripts/experiments/run_multiple_choice.py \
    --model_type $MODEL_TYPE \
    --task_name exams \
    --do_test \
    --para_type per_choice \
    --model_name_or_path $TRAINED_MODEL_DIR \
    --data_dir $INPUT_DATA_DIR \
    --max_seq_length $MAX_SEQ_LENGTH \
    --output_dir $OUTPUT_DIR \
    --per_gpu_eval_batch_size=$EVAL_BATCH_SIZE \
    --overwrite_cache \
    --overwrite_output
``````

