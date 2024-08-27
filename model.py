import timm
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, model_name: str = "resnet50", n_classes=2, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, in_chans=1, drop_path_rate=0.2
        )
        self.backbone.reset_classifier(0, "avg")

        self.linear = nn.Linear(self.backbone.num_features, n_classes)

    def feature(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
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
            loss = F.cross_entropy(logits, labels)

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


