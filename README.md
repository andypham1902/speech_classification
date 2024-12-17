### Commands

```
```bash

torchrun  --nproc_per_node=4 --master_port=10001 train.py --output_dir outputs --do_train --do_eval --remove_unused_columns False --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --learning_rate 1e-4 --warmup_ratio 0.01 --lr_scheduler_type cosine --save_strategy epoch --evaluation_strategy epoch --logging_strategy steps --logging_steps 20 --save_total_limit 2 --load_best_model_at_end True --fp16 --optim adamw_torch --weight_decay 1e-2 --num_train_epochs 20 --metric_for_best_model eval_accuracy --greater_is_better=True --dataloader_num_workers=32 --max_grad_norm=1.0 --overwrite_output_dir=True --report_to none --model_name tf_efficientnetv2_s_in21ft1k --train_data_file emotions.csv 
```

```
python inference.py \
    --resume_from_checkpoint /path/to/your/checkpoint \
    --train_data_file /path/to/your/data.csv \
    --fold 0 \
    --model_name_or_path tf_efficientnetv2_s_in21ft1k
```

External data: https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition/notebook