import timm
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, model_name: str = "resnet50", n_classes=2, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, in_chans=1, drop_path_rate=0.3
        )
        self.backbone.reset_classifier(0, "avg")
        
        # Add temporal attention
        self.attention = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(self.backbone.num_features, n_classes)

    def feature(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        
        # Apply attention and dropout
        attention_weights = self.attention(features)
        features = features + features * attention_weights
        features = self.dropout(features)
        
        features = self.linear(features)
        return features

    @torch.amp.autocast("cuda")
    def forward(
        self, 
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ):
        logits = self.feature(images)
        loss = None
        if labels is not None:
            # Create class weights tensor - give more weight to neutral class
            class_weights = torch.ones(8, device=logits.device)
            class_weights[5] = 20.0  # Increase weight for neutral class (index 5)
            class_weights[2] = 0.5   # Decrease weight for disgust class (index 2)
            class_weights = class_weights / class_weights.sum()
            
            # Add weighted cross entropy with label smoothing
            loss = F.cross_entropy(
                logits, 
                labels, 
                weight=class_weights,
                label_smoothing=0.1
            )

        if return_dict:
            return {"loss": loss, "logits": logits}

        return (loss, logits)
    
if __name__ == "__main__":
    from dataset import QuestionDataset, collate_fn

    model = Model("tf_efficientnetv2_s.in21k_ft_in1k")
    model = model.cuda()
    import pandas as pd
    df = pd.read_csv("data/train_question.csv")
    dataset = QuestionDataset(df)
    batch = collate_fn([dataset[10], dataset[30]])
    for k, v in batch.items():
        batch[k] = v.cuda()

    loss, outputs = model(**batch)
    import pdb;pdb.set_trace()


