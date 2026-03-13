from model import DualWindowAttModel
from dataset import AffWild2WindowDataset
import torch
import os
import numpy as np
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import torch.nn as nn
from scipy.signal import medfilt
from datasets import load_from_disk

D_M = 256
NUM_LAYER = 3
MDROP =0 #default =0.1
OUTPUT_DIR = f"dual_window_att_model/checkpoints"
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")
EPOCHS = 30
BATCH_SIZE = 32
L_R = 2.4e-5


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        
        # logits: (Batch, Window, 8) -> (Batch * Window, 8)
        # labels: (Batch, Window) -> (Batch * Window)
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)

        # 计算标准交叉熵，遇到 ignore_index 会自动输出 0
        ce_loss = nn.functional.cross_entropy(
            logits,
            labels,
            reduction="none",
            weight=self.weight,
            ignore_index=self.ignore_index,
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # 只对有效的帧求平均
        valid_mask = labels != self.ignore_index
        if valid_mask.sum() > 0:
            return focal_loss[valid_mask].mean()
        else:
            return focal_loss.sum() * 0.0  # 防止整个 batch 都是 -1 导致 NaN


def predict_single_video(
    model,
    v_feat_full,
    a_feat_full,
    v_mask_full,
    window_size=64,
    stride=8,
    device="cuda",
):
    """
    对单个长视频进行滑动窗口软投票推理，返回全长预测结果。
    """
    model.eval()
    seq_len = v_feat_full.shape[0]
    num_classes = 8

    sum_logits = torch.zeros((seq_len, num_classes), device=device)
    count = torch.zeros((seq_len, 1), device=device)

    v_feat_full = torch.FloatTensor(v_feat_full).to(device)
    a_feat_full = torch.FloatTensor(a_feat_full).to(device)
    v_mask_full = torch.LongTensor(v_mask_full).to(device)

    # 正常滑动窗口推理
    for start in range(0, seq_len, stride):
        end = start + window_size
        if end > seq_len:
            # 边界处理：凑齐最后一个 window
            start = seq_len - window_size
            end = seq_len

        v_feat_w = v_feat_full[start:end].unsqueeze(0)
        a_feat_w = a_feat_full[start:end].unsqueeze(0)
        v_mask_w = v_mask_full[start:end].unsqueeze(0)

        with torch.no_grad():
            logits = model(v_feat_w, a_feat_w, v_mask_w)

        sum_logits[start:end] += logits.squeeze(0)
        count[start:end] += 1

        if end == seq_len:
            break

    # 软投票计算平均 Logits，并输出预测
    avg_logits = sum_logits / count.clamp(min=1.0)
    frame_level_preds = torch.argmax(avg_logits, dim=-1).cpu().numpy()

    return frame_level_preds


from tqdm import tqdm  

class CustomTrainer(Trainer):
    def __init__(
        self,
        raw_val_dataset=None,
        class_weights=None,
        gamma=2.0,
        visual_dropout_prob=MDROP,  
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_val_dataset = raw_val_dataset
        self.class_weights = class_weights
        self.gamma = gamma
        self.visual_dropout_prob = visual_dropout_prob  

        self.loss_fct = FocalLoss(
            weight=self.class_weights, gamma=self.gamma, ignore_index=-1
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        
        if model.training and self.visual_dropout_prob > 0:
            # inputs["v_mask"] shape: (Batch, Window)
        
            batch_size = inputs["v_mask"].size(0)

            # 小于 prob 的样本将被 Mask
            dropout_mask = (
                torch.rand(batch_size, device=inputs["v_mask"].device)
                < self.visual_dropout_prob
            )

            if dropout_mask.any():
                # 将命中样本的mask 置为 0 
                inputs["v_mask"][dropout_mask, :] = 0

        labels = inputs.pop("labels")

        logits = model(**inputs)

        if (
            self.class_weights is not None
            and self.loss_fct.weight.device != logits.device
        ):
            self.loss_fct.weight = self.class_weights.to(logits.device)

        loss = self.loss_fct(logits, labels)
        return (loss, (loss, logits)) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self.model.eval()
        all_preds = []
        all_labels = []

        # 遍历原始视频集
        for row in tqdm(self.raw_val_dataset, desc="Evaluating Videos"):
            v_feat = row["visual_feat"]
            a_feat = row["audio_feat"]
            v_mask = row["v_mask"]
            labels = np.array(row["label"])

            preds = predict_single_video(
                self.model,
                v_feat,
                a_feat,
                v_mask,
                window_size=64,
                stride=8,
                device=self.args.device,
            )

            # 时序平滑后处理
            preds = medfilt(preds, kernel_size=11)

            preds = preds.astype(np.int64)
            # 只保留有效标签 
            valid_idx = labels != -1
            if valid_idx.sum() > 0:
                all_preds.extend(preds[valid_idx])
                all_labels.extend(labels[valid_idx])

        # 全局指标计算
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")

        metrics = {
            f"{metric_key_prefix}_accuracy": accuracy,
            f"{metric_key_prefix}_macro_f1": f1,
        }

        print(f"\n[Validation] Accuracy: {accuracy:.4f} | Macro F1: {f1:.4f}")
        print("=" * 50 + "\n")

        self.log(metrics)


        return metrics


def main():
    model = DualWindowAttModel(d_model=D_M, num_layers=NUM_LAYER)
    train_data = AffWild2WindowDataset(
        hf_dataset_path="features/multimodal_feature/beit-large/train",
        window_size=64,
        stride=8,
    )
    class_weights = train_data.get_class_weights(method="effective_num")

    raw_val_dataset = load_from_disk(
        "features/multimodal_feature/beit-large/val"
    ).with_format("numpy")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=os.path.join(OUTPUT_DIR, "log"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        learning_rate=L_R,
        weight_decay=0.03,
        lr_scheduler_type="cosine",  
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        bf16=True,  
        dataloader_num_workers=4,  
        logging_strategy="epoch",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )

    
    trainer = CustomTrainer(
        raw_val_dataset=raw_val_dataset,  # 传入未切片的原始数据
        class_weights=class_weights,
        gamma=2.0,
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=train_data,  
    )


    print("开始训练...")
    trainer.train()

    print("训练结束...,加载最优模型保存")
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    trainer.save_model(BEST_MODEL_DIR)

if __name__ == "__main__":
    main()
