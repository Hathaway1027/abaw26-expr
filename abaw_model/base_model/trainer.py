import os
import numpy as np
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import numpy as np
from datasets import load_from_disk
from basemodel import BaseModel


train_dataset = load_from_disk(
    "features/multimodal_feature/beit-large/train_frame"
)
val_dataset = load_from_disk(
    "features/multimodal_feature/beit-large/val_frame"
)

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")


OUTPUT_DIR = "base_model/checkpoints"
LOG_DIR = "base_model/checkpoints/logs"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model")

BATCH_SIZE = 512
EPOCH = 50
L_R = 1e-4

counts = {
    0: 177198,
    1: 16573,
    2: 10771,
    3: 9080,
    4: 95463,
    5: 78751,
    6: 31615,
    7: 165866,
}
class_counts = torch.tensor([counts[i] for i in range(8)], dtype=torch.float)


weights = 1.0 / torch.sqrt(class_counts)
DATA_WEIGHT = weights / weights.mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        
        ce_loss = nn.functional.cross_entropy(
            logits,
            labels,
            reduction="none",
            weight=self.weight,
        )

        
        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def compute_metrics(p):
    predictions = np.argmax(p.predictions, axis=1)
    labels = p.label_ids

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")

    return {"accuracy": accuracy, "f1": f1}


class CustomTrainer(Trainer):

    def __init__(
        self,
        class_weights=None,
        gamma=2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.gamma = gamma
        
        self.loss_fct = FocalLoss(weight=self.class_weights, gamma=self.gamma)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if (
            self.class_weights is not None
            and self.loss_fct.weight.device != logits.device
        ):
            self.loss_fct.weight = self.class_weights.to(logits.device)

        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def multimodal_collator(features):
    
    batch = {}
    
    batch["visual_feat"] = torch.stack(
        [torch.tensor(f["visual_feat"]) for f in features]
    )
    
    batch["audio_feat"] = torch.stack([torch.tensor(f["audio_feat"]) for f in features])
    
    batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
    return batch


def main(lambda_val):
    # 修改λ值控制融合权重，观察多模态融合的作用
    model = BaseModel(lambda_val=lambda_val)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,  
        num_train_epochs=EPOCH,  
        per_device_train_batch_size=BATCH_SIZE, 
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=L_R,  
        logging_dir=LOG_DIR,
        logging_strategy="epoch",
        eval_strategy="epoch",  
        save_strategy="epoch",  
        load_best_model_at_end=True,  
        metric_for_best_model="f1",
        bf16=True,
        dataloader_pin_memory=True,
        report_to="tensorboard",
        save_total_limit=5,  
        remove_unused_columns=False,  
        dataloader_num_workers=4,  
        ddp_find_unused_parameters=False,
        label_names=["labels"],
    )

    trainer = CustomTrainer(
        class_weights=DATA_WEIGHT,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=multimodal_collator,
    )


    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    trainer.train()

    
    print("Training finished. Best model is loaded and now being saved...")
    os.makedirs(BEST_MODEL_PATH, exist_ok=True)
    trainer.save_model(BEST_MODEL_PATH)

if __name__ == "__main__":
    lambda_val = [0, 0.5, 0.7, 1]
    for lv in lambda_val:
        main(lv)
