import gc
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import Tensor, nn
from torch.nn import CosineEmbeddingLoss
from torchvision.ops import sigmoid_focal_loss
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach

from dataset import RatioSampler
from model import Model
from samplers import ProportionalTwoClassesBatchSampler
import utils


class CustomTrainer(Trainer):
    def __init__(self, pos_neg_ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.pos_neg_ratio = pos_neg_ratio
        self.loss_fn = nn.CrossEntropyLoss()

    def _get_train_sampler(self):
        return RatioSampler(
            dataset=self.train_dataset,
            ratio=self.pos_neg_ratio,
        )
        # pos_bsize = self.args.train_batch_size // (self.pos_neg_ratio + 1)
        # return ProportionalTwoClassesBatchSampler(
        #     np.array(self.train_dataset.labels, dtype=np.int64),
        #     self.args.train_batch_size,
        #     minority_size_in_batch=pos_bsize,
        # )

    def compute_loss(self, model: Model, inputs: Dict, return_outputs=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        outputs = model(inputs["images"])
        loss = self.loss_fn(outputs, inputs["labels"].long().to(device))

        if return_outputs:
            return (loss, outputs)
        return loss

    def create_optimizer(self):
        model = self.model
        no_decay = []
        for n, m in model.named_modules():
            if isinstance(
                m,
                (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.LayerNorm,
                    torch.nn.LayerNorm,
                    torch.nn.GroupNorm,
                ),
            ):
                no_decay.append(n)

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def prediction_step(
        self, model, inputs, prediction_loss_only=False, ignore_keys=None
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        gc.collect()
        return loss, outputs, inputs["labels"]


def compute_metrics(eval_preds):
    # calculate accuracy using sklearn's function
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    softmax_preds = F.softmax(torch.tensor(logits), dim=-1).numpy()
    solution_data = {
        'id': list(range(len(labels))),
        'label': labels
    }
    
    submission_data = {
        'id': list(range(len(labels))),
        'score': softmax_preds[:, 1].tolist()
    }
    
    solution_df = pd.DataFrame(solution_data)
    submission_df = pd.DataFrame(submission_data)

    # Define the row ID column name and min TPR
    row_id_column_name = 'id'
    min_tpr = 0.80

    # Calculate the pAUC score
    pauc = utils.score(solution_df, submission_df, row_id_column_name, min_tpr)
        
    return {
        "accuracy": accuracy_score(y_true=labels, y_pred=predictions),
        "f1": f1_score(y_true=labels, y_pred=predictions, average="macro"),
        "pauc": pauc,
    }
