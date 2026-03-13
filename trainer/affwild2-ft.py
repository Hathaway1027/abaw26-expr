import os
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    TrainingArguments,
    Trainer,
    ViTForImageClassification,
    EarlyStoppingCallback,
    BeitForImageClassification,
)
from datasets import load_dataset
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
    RandomHorizontalFlip,
    RandomGrayscale,
    InterpolationMode,
)
from sklearn.metrics import accuracy_score, f1_score


MODEL_NAME = ["mae-affwild2", "beit-affwild2"]
NUM_CLASSES = 8

OUTPUT_DIR = {
    "mae-affwild2": "models/mae-affwild2",
    "beit-affwild2": "models/beit-affwild2",
}

BEST_MODEL_PATH = {
    "mae-affwild2": "models/best_mae_affwild2",
    "beit-affwild2": "models/best_beit_affwild2",
}

MODEL_PATH = {
    "mae-affwild2": "models/best_mae_ft",
    "beit-affwild2": "models/beit-large",
}

MODEL = {
    "mae-affwild2": ViTForImageClassification.from_pretrained(
        MODEL_PATH["mae-affwild2"],
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ),
    "beit-affwild2": BeitForImageClassification.from_pretrained(
        MODEL_PATH["beit-affwild2"],
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ),
}

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

BATCH_SIZE = {"mae-affwild2": 256, "beit-affwild2": 64}


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        
        ce_loss = nn.functional.cross_entropy(
            logits, labels, reduction="none", weight=self.weight
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


class WeightedTrainer(Trainer):

    def __init__(
        self,
        class_weights=None,
        gamma=2.0,
        backbone_lr=None,
        classifier_lr=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.gamma = gamma
       
        self.loss_fct = FocalLoss(weight=self.class_weights, gamma=self.gamma)
        self.backbone_lr = backbone_lr
        self.classifier_lr = classifier_lr

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs

        if (
            self.class_weights is not None
            and self.loss_fct.weight.device != logits.device
        ):
            self.loss_fct.weight = self.class_weights.to(logits.device)

        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        
        if self.optimizer is None:

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "classifier" not in n and p.requires_grad
                    ],
                    "lr": self.backbone_lr,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "classifier" in n and p.requires_grad
                    ],
                    "lr": self.classifier_lr,
                    "weight_decay": 0.01,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

        return self.optimizer

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        
        if self.lr_scheduler is None:
            
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer if optimizer is not None else self.optimizer,
                T_max=num_training_steps,
            )
        return self.lr_scheduler


def main(model_name):
    
    LEARNING_RATE = 4e-5
    EPOCHS = 50

    model = MODEL[model_name]

    transforms = Compose(
        [
            Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(),
            # RandomGrayscale(p=0.3),
            ToTensor(),
            Normalize(mean=MEAN, std=STD),
        ]
    )

    def preprocess_images(examples):

        return {
            "pixel_values": [
                transforms(img.convert("RGB")) for img in examples["image"]
            ],
            "labels": [label for label in examples["label"]],
        }

    print("Loading datasets...")

    data_files = {
        "train": os.path.join(
            "dataset/Affwild2_ft",
            "train.parquet",
        ),
        "validation": os.path.join(
            "dataset/Affwild2_ft",
            "val.parquet",
        ),
    }
    raw_datasets = load_dataset("parquet", data_files=data_files)

    train_dataset = raw_datasets["train"]
    val_dataset = raw_datasets["validation"]

    train_dataset.set_transform(preprocess_images)
    val_dataset.set_transform(preprocess_images)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    os.makedirs(OUTPUT_DIR[model_name], exist_ok=True)
    print(f"Model save path: {OUTPUT_DIR[model_name]}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR[model_name],
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE[model_name],
        per_device_eval_batch_size=BATCH_SIZE[model_name],
        warmup_ratio=0.1,
        learning_rate=LEARNING_RATE,
        weight_decay=1e-4,
        logging_dir=f"{OUTPUT_DIR[model_name]}/logs",
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        bf16=True,
        gradient_accumulation_steps=8,
        dataloader_num_workers=(256//BATCH_SIZE[model_name]),
        report_to="tensorboard",
        remove_unused_columns=False,
        label_smoothing_factor=0.1,
        ddp_find_unused_parameters=False,
    )
    print(f"Training arguments initialized")

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
    weights = weights / weights.mean()

    trainer = WeightedTrainer(
        class_weights=weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        backbone_lr=LEARNING_RATE,
        classifier_lr=LEARNING_RATE * 20,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=10, early_stopping_threshold=0.0
            )
        ],
    )
    print(f"WeightedTrainer created successfully")

    print("Starting training ...")
    trainer.train()

    print("Training finished. Best model is loaded and now being saved...")
    os.makedirs(BEST_MODEL_PATH[model_name], exist_ok=True)
    trainer.save_model(BEST_MODEL_PATH[model_name])

if __name__ == "__main__":
    
    main("beit-affwild2")
