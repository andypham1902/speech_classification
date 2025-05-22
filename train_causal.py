import argparse
import hashlib
import math
import os
import sys
from functools import partial
from time import time

import accelerate
import datasets
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import Dataset, disable_progress_bar
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import (AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed)


datasets.disable_caching()
disable_progress_bar()
set_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default="train", type=str)
parser.add_argument("--model_path", default="google/gemma-2-9b-it", type=str)
parser.add_argument("--max_length", default=8192, type=int)
parser.add_argument("--learning_rate", default=2e-6, type=float)
parser.add_argument("--adam_beta2", default=0.999, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--per_device_train_batch_size", default=2, type=int)
parser.add_argument("--per_device_eval_batch_size", default=2, type=int)
parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower() == "true", default=True)
parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
parser.add_argument("--datasets_dir", default="data", type=str)
parser.add_argument("--dataset_name", default="new_train_0fold", type=str)
parser.add_argument("--valid_name", default="new_valid_0fold", type=str)
parser.add_argument("--output_dir", default="ckpts", type=str)
parser.add_argument("--eval_steps", default=1000, type=int)
parser.add_argument("--scheduler", default="linear", type=str)
parser.add_argument("--warmup_steps", default=50, type=int)
parser.add_argument("--testing", type=lambda x: x.lower() == "true", default=False)
parser.add_argument("--num_epochs", default=20, type=int, help="Number of epochs to train for")
args = parser.parse_args(args=[] if "__file__" not in globals() else sys.argv[1:])


def linear_with_min_lr(current_step: int, *, num_warmup_steps: int, num_training_steps: int, min_lr: float, max_lr: float):
    """
    Linear schedule:
      1. Warm up from min_lr to max_lr over [0, num_warmup_steps].
      2. Then decay from max_lr back to min_lr over [num_warmup_steps, num_training_steps].
    """
    current_step = max(0, min(current_step, num_training_steps))

    # Warmup phase: from min_lr to max_lr
    if current_step < num_warmup_steps:
        ratio = current_step / max(1, num_warmup_steps)
        return min_lr + ratio * (max_lr - min_lr)
    # Decay phase: from max_lr back to min_lr
    else:
        # How far we are into the decay phase, from 0.0 to 1.0
        ratio = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max_lr - ratio * (max_lr - min_lr)

def fmt(truncate_side, sentence, max_length, tokenizer):
    sentence = sentence.strip()
    # Create a structured message for classification
    messages = [
        {"role": "user", "content": f"/no_think Classify this sentence into: Happy, Sad, Neutral.\n{sentence}"},
        {"role": "assistant", "content": ""}  # Add empty assistant message to position the model for generation
    ]

    # Apply the model's chat template
    if hasattr(tokenizer, 'apply_chat_template'):
        formatted_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback for tokenizers without chat template
        formatted_text = "/no_think Classify this sentence into: Happy, Sad, Neutral.\n" + sentence + "\nAssistant: "

    # Tokenize the formatted text
    tokenized = tokenizer(formatted_text, truncation=True, max_length=max_length, 
                          padding="max_length", return_tensors="pt")
    input_ids = tokenized["input_ids"].squeeze(0)
    attention_mask = tokenized["attention_mask"].squeeze(0)

    # Ensure the input_ids and attention_mask are of the correct length
    if truncate_side == "left":
        input_ids = input_ids[-max_length:]
        attention_mask = attention_mask[-max_length:]
    else:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "formatted_text": formatted_text}


def main():
    accelerator = Accelerator(log_with="wandb", gradient_accumulation_steps=args.gradient_accumulation_steps)
    print0 = accelerator.print

    global_bs = args.per_device_train_batch_size*args.gradient_accumulation_steps*int(os.environ.get("WORLD_SIZE", "8"))
    experiment = args.experiment + "_" if args.experiment != "train" else ""

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.add_eos_token = True
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"  # For causal LM

    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer does not have pad token, please add pad token to tokenizer.")

    # Initialize model
    print0("Using default ForCausalLM model")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.to('cuda')
    happy_token = tokenizer("Happy", return_tensors="pt").input_ids.item()
    sad_token = tokenizer("Sad", return_tensors="pt").input_ids.item()
    neutral_token = tokenizer("Neutral", return_tensors="pt").input_ids.item()
    print0(f"Happy token: {happy_token}, Sad token: {sad_token}, Neutral token: {neutral_token}")

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Load datasets
    print0("Loading datasets...")
    train_ds = Dataset.from_parquet(f"{args.datasets_dir}/{args.dataset_name}.parquet")
    valid_ds = Dataset.from_parquet(f"{args.datasets_dir}/{args.valid_name}.parquet")

    if args.testing:
        train_ds = train_ds.select(range(0, 100))
        valid_ds = valid_ds.select(range(0, 100))

    # Setup run name and output directory
    run_name = f"{experiment}lm_{args.dataset_name}_{args.model_path.split('/')[-1]}_lr{args.learning_rate}_bs{global_bs}_{args.valid_name}"
    max_length = args.max_length
    args.run_name = run_name
    args.output_dir = f"{args.output_dir}/{run_name}"

    print0(f"Running experiment: {run_name}")
    print0(f"Output directory: {args.output_dir}")

    if os.path.isdir(args.output_dir) and len(os.listdir(args.output_dir)) > 0:
        print0(f"Output directory non-empty, this experiment has likely run already, exiting...")
        return

    accelerator.init_trackers(project_name="whitefebruary", config=vars(args), init_kwargs={"wandb": {"name": run_name}})

    # Format datasets
    def format_ds(ds, max_length): 
        ds = ds.map(
            lambda sample: fmt(
                truncate_side="left",  # Match tokenizer.truncation_side 
                sentence=sample["sentence"] if "sentence" in sample else str(sample),
                max_length=max_length, 
                tokenizer=tokenizer
            ),
            batch_size=10000
        )

        # Convert to multi-class classification (3 classes)
        def determine_emotion(sample):
            # Check if dataset already has emotion labels
            if 'emotion' in sample:
                emotion = sample['emotion'].lower()
                if 'happy' in emotion or 'joy' in emotion:
                    return 0  # happy
                elif 'sad' in emotion or 'negative' in emotion:
                    return 1  # sad
                else:
                    return 2  # neutral           
            # Default to neutral if we can't determine
            return 2  # neutral

        ds = ds.map(lambda x: {"labels": determine_emotion(x)})
        return ds

    train_ds = format_ds(train_ds, max_length=max_length)
    valid_ds = format_ds(valid_ds, max_length=128)

    print0(f"Length of train dataset: {len(train_ds)}")
    print0(f"Length of valid dataset: {len(valid_ds)}")

    # Define collate functions
    def collate_fn(batch):
        return {k: [x[k] for x in batch] for k in batch[0].keys()}

    # Create dataloaders
    dataloader = DataLoader(train_ds, batch_size=args.per_device_train_batch_size, collate_fn=collate_fn, shuffle=True)
    eval_names = ["val"]
    eval_dataloaders = [
        DataLoader(valid_ds, batch_size=args.per_device_eval_batch_size, collate_fn=collate_fn, shuffle=False),
    ]

    # Setup optimizer and learning rate scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, args.adam_beta2))
    model, opt, dataloader, *eval_dataloaders = accelerator.prepare(model, opt, dataloader, *eval_dataloaders)
    steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    num_training_steps = steps_per_epoch * args.num_epochs * 8

    if args.scheduler == "linear_min":
        scheduler = LambdaLR(opt, partial(linear_with_min_lr, min_lr=0.1, max_lr=1.0, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps), last_epoch=-1)
    else:
        scheduler = get_scheduler(args.scheduler, optimizer=opt, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
    scheduler = accelerator.prepare(scheduler)

    model.train()
    step = 0
    best_accuracy = 0.0

    def evaluate(model):
        model.eval()
        nonlocal best_accuracy

        for eval_name, eval_dataloader in zip(eval_names, eval_dataloaders):
            total_loss_local = 0.0
            total_correct_local = 0
            total_samples_local = 0
            # Track per-class accuracy
            class_correct_local = [0, 0, 0]  # happy, sad, neutral
            class_total_local = [0, 0, 0]

            for batch in tqdm(eval_dataloader, desc=f"Evaluating on {eval_name}", disable=not accelerator.is_main_process):
                with torch.no_grad():
                    padded = tokenizer.pad({"input_ids": batch["input_ids"]}, return_tensors="pt")
                    input_ids = padded["input_ids"].to(model.device)
                    attn_mask = padded["attention_mask"].to(model.device)

                    base_model = model.module if hasattr(model, "module") else model
                    # Get logits from the model
                    hs = base_model.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True).hidden_states[-1][:, -1, :]
                    logits = base_model.lm_head(hs)
                    logits = logits[:, [happy_token, sad_token, neutral_token]]
                    
                    loss = nn.CrossEntropyLoss()(logits, torch.tensor(batch["labels"]).to(model.device))
                    probs = logits.cpu().softmax(-1)

                    preds = probs.argmax(-1).cpu().numpy()
                    labels = np.array(batch["labels"])

                    correct = preds == labels

                    # Track per-class accuracy
                    for i, (pred, label) in enumerate(zip(preds, labels)):
                        if label < 3:  # Ensure valid class index
                            class_total_local[label] += 1
                            if pred == label:
                                class_correct_local[label] += 1

                    total_samples_local += preds.shape[0]
                    total_loss_local += loss.item() * input_ids.shape[0]
                    total_correct_local += correct.sum()

            total_loss, total_correct, total_samples = [
                accelerator.gather(torch.tensor(x).to(model.device)).sum().item()
                for x in (total_loss_local, total_correct_local, total_samples_local)
            ]

            # Gather per-class metrics
            class_correct = [accelerator.gather(torch.tensor(x).to(model.device)).sum().item() for x in class_correct_local]
            class_total = [accelerator.gather(torch.tensor(x).to(model.device)).sum().item() for x in class_total_local]

            accuracy = total_correct / total_samples
            loss = total_loss / total_samples

            # Calculate per-class accuracy
            class_accuracy = [
                class_correct[i] / max(class_total[i], 1) for i in range(3)
            ]

            print0(f"[{eval_name}] accuracy: {accuracy:.4f}, loss: {loss:.4f}, total_samples: {total_samples}")
            print0(f"[{eval_name}] Per-class accuracy - Happy: {class_accuracy[0]:.4f}, Sad: {class_accuracy[1]:.4f}, Neutral: {class_accuracy[2]:.4f}")

             # Save model if accuracy improves (using the first eval dataset as the metric)
            if eval_name == eval_names[0] and accuracy > best_accuracy:
                prev_best = best_accuracy
                best_accuracy = accuracy

                # Save the model
                best_model_dir = os.path.join(args.output_dir, f"best_checkpoint_{accuracy:.4f}")
                os.makedirs(best_model_dir, exist_ok=True)

                print0(f"New best accuracy: {best_accuracy:.4f} (previous: {prev_best:.4f})")
                print0(f"Saving checkpoint to {best_model_dir}")

                # Save the model
                accelerator.unwrap_model(model).save_pretrained(
                    best_model_dir,
                    save_function=accelerator.save,
                    is_main_process=accelerator.is_main_process,
                    state_dict=accelerator.get_state_dict(model),
                )

                if accelerator.is_main_process:
                    tokenizer.save_pretrained(best_model_dir)
                    if os.path.exists(best_model_dir + "/model.safetensors.index.json") and os.path.exists(best_model_dir + "/model.safetensors"):
                        os.remove(best_model_dir + "/model.safetensors")

            accelerator.log({
                f"eval/{eval_name}-accuracy": accuracy,
                f"eval/{eval_name}-loss": loss,
                f"eval/{eval_name}-acc-happy": class_accuracy[0],
                f"eval/{eval_name}-acc-sad": class_accuracy[1],
                f"eval/{eval_name}-acc-neutral": class_accuracy[2]
            }, step=step)

        model.train()

    # Training loop
    for epoch in range(args.num_epochs):
        print0(f"Starting epoch {epoch+1}/{args.num_epochs}")
        tbar = trange(math.ceil(len(dataloader) / args.gradient_accumulation_steps), disable=not accelerator.is_main_process)
        for batch in dataloader:
            if step % args.eval_steps == 0 and accelerator.sync_gradients:
                evaluate(model)

            with accelerator.accumulate(model):
                stime = time()
                padded = tokenizer.pad({"input_ids": batch["input_ids"]}, return_tensors="pt")
                input_ids = padded["input_ids"].to(model.device)
                attn_mask = padded["attention_mask"].to(model.device)

                # Forward pass through causal LM
                base_model = model.module if hasattr(model, "module") else model
                hs = base_model.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True).hidden_states[-1][:, -1, :]
                logits = base_model.lm_head(hs)
                logits = logits[:, [happy_token, sad_token, neutral_token]]

                # Calculate loss
                loss = nn.CrossEntropyLoss()(logits, torch.tensor(batch["labels"]).to(model.device))

                # Backward pass
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    opt.step()
                    opt.zero_grad()
                    scheduler.step()

                    loss = accelerate.utils.reduce(loss).item()
                    forward_time = time() - stime

                    tbar.update(1)
                    tbar.set_description(f"Loss: {loss:.4f}")
                    accelerator.log({"train/loss": loss, "train/learning_rate": float(scheduler.get_last_lr()[0]), "train/forward_time": forward_time}, step=step)
                    step += 1

    # Final evaluation
    evaluate(model)

    # Save model and tokenizer
    accelerator.unwrap_model(model).save_pretrained(
        args.output_dir,
        save_function=accelerator.save,
        is_main_process=accelerator.is_main_process,
        state_dict=accelerator.get_state_dict(model),
    )

    if accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)
        if os.path.exists(args.output_dir + "/model.safetensors.index.json") and os.path.exists(args.output_dir + "/model.safetensors"):
            os.remove(args.output_dir + "/model.safetensors")

    accelerator.print(f"Checkpoint: {args.output_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
