/home/andrew/miniconda3/envs/py312/bin/accelerate launch --num_processes 8 train_causal.py \
--model_path Qwen/Qwen3-0.6B \
--max_length 128 \
--learning_rate 8e-5 \
--dataset_name text_emotion_train \
--valid_name text_emotion_validation \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 1 \
--eval_steps 20 