from contextlib import contextmanager

import logging
import os
import sys
import safetensors.torch
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import transformers
from transformers import TrainingArguments, set_seed, Trainer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer import logger
from sklearn.metrics import accuracy_score, f1_score
from dataset import QuestionDataset, collate_fn, EmotionDataset
from model import Model
from configs import (
    H4ArgumentParser,
    DataArguments,
    ModelArguments,
)


@contextmanager
def distributed_barrier(rank, is_initialized):
    if rank > 0:
        logger.info("Waiting for the main process ...")
        dist.barrier()
    try:
        yield
    finally:
        if rank == 0 and is_initialized:
            logger.info("Loading results from the main process ...")
            dist.barrier()

def compute_metrics(eval_preds):
    # calculate accuracy using sklearn's function
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
        
    return {
        "accuracy": accuracy_score(y_true=labels, y_pred=predictions),
        "f1": f1_score(y_true=labels, y_pred=predictions, average="macro"),
    }

def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    do_eval: bool = training_args.do_eval

    # Set seed
    set_seed(training_args.seed)

    # Setup logging
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    df = pd.read_csv(data_args.train_data_file)
    df_val = pd.read_csv(data_args.eval_data_file)
    # Load datasets
    # train_dataset = QuestionDataset(
    #     df[df.fold != data_args.fold],
    #     mode="train",
    # )
    # if data_args.max_samples > 0:
    #     train_dataset = train_dataset.select(
    #         range(data_args.max_samples)
    #     )

    # valid_dataset = QuestionDataset(
    #     df[df.fold == data_args.fold],
    #     mode="val",
    # )
    train_dataset = EmotionDataset(
        df,
        # df[df.fold != data_args.fold],
        mode="train",
        # sr=16000,
    )
    if data_args.max_samples > 0:
        train_dataset = train_dataset.select(
            range(data_args.max_samples)
        )

    valid_dataset = EmotionDataset(
        # df[df.fold == data_args.fold],
        df_val,
        mode="val",
        # sr=16000,
    )
    # Initialize trainer
    print("Initializing model...")
    model = Model(
        model_name=model_args.model_name_or_path,
        n_classes=8,
    )

    print("Start training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and not training_args.overwrite_output_dir
        ):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is not None:
                logger.info(
                    f"Resuming training from last checkpoint: {last_checkpoint}"
                )
                trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            if training_args.resume_from_checkpoint is not None:
                logger.info(
                    f"Resuming from checkpoint: {training_args.resume_from_checkpoint}"
                )
                trainer.model.load_state_dict(
                    safetensors.torch.load_file(training_args.resume_from_checkpoint)
                )
            trainer.train()

        if training_args.deepspeed and is_deepspeed_zero3_enabled():
            trainer.accelerator.wait_for_everyone()
            state_dict = trainer.accelerator.get_state_dict(trainer.model_wrapped)
            if trainer.is_world_process_zero():
                trainer.accelerator.save(
                    state_dict,
                    os.path.join(training_args.output_dir, "model.safetensors"),
                )
        else:
            trainer.save_model()

    if do_eval:
        trainer.model.load_state_dict(
            safetensors.torch.load_file(training_args.output_dir + "/model.safetensors")
        )
        outputs = trainer.predict(valid_dataset)

        if trainer.is_world_process_zero():
            print(outputs.metrics)
            trainer.save_metrics("eval", outputs.metrics)
            np.save(
                training_args.output_dir + "/eval_predictions.npy",
                outputs.predictions,
            )


if __name__ == "__main__":
    main()
