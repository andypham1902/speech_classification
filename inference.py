import os
import numpy as np
import pandas as pd
import torch
import safetensors.torch
from transformers import set_seed

from dataset import EmotionDataset, collate_fn
from model import Model
from configs import (
    H4ArgumentParser,
    DataArguments,
    ModelArguments,
)
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score, f1_score

def main():
    # Parse arguments
    parser = H4ArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load validation data
    df = pd.read_csv(data_args.train_data_file)
    valid_dataset = EmotionDataset(
        df[df.fold == data_args.fold],
        mode="val",
    )
    
    # Initialize model
    print("Loading model...")
    model = Model(
        model_name=model_args.model_name_or_path,
        n_classes=8,
    )
    
    # Load checkpoint
    checkpoint_path = os.path.join(model_args.resume_from_checkpoint, "model.safetensors")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(safetensors.torch.load_file(checkpoint_path))
    else:
        raise ValueError(f"Checkpoint not found at {checkpoint_path}")
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Run inference
    print("Running inference...")
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(images=batch["images"], labels=None, return_dict=True)
            predictions = torch.softmax(outputs["logits"], dim=-1)
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
    
    # Concatenate all predictions
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate accuracy and F1 score
    pred_labels = np.argmax(all_predictions, axis=1)
    accuracy = accuracy_score(all_labels, pred_labels)
    f1 = f1_score(all_labels, pred_labels, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save predictions
    output_dir = model_args.resume_from_checkpoint
    output_path = os.path.join(output_dir, "predictions.npy")
    label_path = os.path.join(output_dir, "labels.npy")
    np.save(output_path, all_predictions)
    np.save(label_path, all_labels)
    print(f"Saved predictions to {output_path}")
    print(f"Saved labels to {label_path}")


if __name__ == "__main__":
    main()
