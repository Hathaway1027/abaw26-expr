import os
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    TrainingArguments,
    Trainer,
    ViTForImageClassification
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

OUTPUT_DIR = "models/mae-ft"
BEST_MODEL_PATH = "models/best_mae_ft"
NUM_CLASSES = 7
MODEL_PATH = "models/best_mae_pt2"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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


def main():
    
    BATCH_SIZE = 256
    LEARNING_RATE = 1.5e-5
    EPOCHS = 100
    
    model = ViTForImageClassification.from_pretrained(MODEL_PATH,num_labels=NUM_CLASSES)
    
    transforms = Compose(
        [
            Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(),
            RandomGrayscale(p=0.3),
            ToTensor(),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
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
        "train": os.path.join("full_dataset", "rafdb_ferplus_affectnet_train.parquet"),
        "validation": os.path.join(
            "full_dataset", "rafdb_ferplus_affectnet_validation.parquet"
        ),
    }
    raw_datasets = load_dataset("parquet", data_files=data_files)

    train_dataset = raw_datasets["train"]
    val_dataset = raw_datasets["validation"]

    train_dataset.set_transform(preprocess_images)
    val_dataset.set_transform(preprocess_images)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Model save path: {OUTPUT_DIR}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=0.1,
        learning_rate=LEARNING_RATE,
        weight_decay=0.05,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        bf16=True,
        gradient_accumulation_steps=2,
        dataloader_num_workers=4,
        report_to="tensorboard",
        remove_unused_columns=False,
        label_smoothing_factor=0.1,
    )
    print(f"Training arguments initialized")

    
    counts = {3: 146733, 4: 87880, 0: 28081, 5: 31017, 6: 18970, 1: 4730, 2: 7319}
    
    class_counts = torch.tensor([counts[i] for i in range(7)], dtype=torch.float)

    weights = 1.0 / torch.sqrt(class_counts)
   
    weights = weights / weights.mean()

    trainer = WeightedTrainer(
        class_weights=weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        backbone_lr=1.5e-4,
        classifier_lr=1e-3,
    )
    print(f"WeightedTrainer created successfully")

    # Train model
    print("Starting training ...")
    trainer.train()

    # Evaluate model
    print("Evaluating model...")
    eval_result = trainer.evaluate()
    print(f"Evaluation results: {eval_result}")

    print("Training finished. Best model is loaded and now being saved...")
    os.makedirs(BEST_MODEL_PATH, exist_ok=True)
    trainer.save_model(BEST_MODEL_PATH)

if __name__ == "__main__":
    main()
