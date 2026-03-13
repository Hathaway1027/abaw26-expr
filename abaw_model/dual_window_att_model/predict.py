from model import DualWindowAttModel
from safetensors.torch import load_model
import torch
from tqdm import tqdm
import numpy as np
from scipy.signal import medfilt
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_from_disk
import os

MODEL_PATH = "dual_window_att_model/checkpoints/best_model/model.safetensors"
DEVICE = "cuda:0"

model = DualWindowAttModel(d_model=256, num_layers=3)
load_model(model, MODEL_PATH)
model.eval()
model.to(DEVICE)

train_dataset = load_from_disk(
    "features/multimodal_feature/beit-large/train"
).with_format("numpy")
val_dataset = load_from_disk(
    "features/multimodal_feature/beit-large/val"
).with_format("numpy")


def predict_single_video(
    model,
    v_feat_full,
    a_feat_full,
    v_mask_full,
    window_size=64,
    stride=8,
    device="cuda",
):
    
    model.eval()
    seq_len = v_feat_full.shape[0]
    num_classes = 8

    sum_logits = torch.zeros((seq_len, num_classes), device=device)
    count = torch.zeros((seq_len, 1), device=device)

    v_feat_full = torch.FloatTensor(v_feat_full).to(device)
    a_feat_full = torch.FloatTensor(a_feat_full).to(device)
    v_mask_full = torch.LongTensor(v_mask_full).to(device)

    for start in range(0, seq_len, stride):
        end = start + window_size
        if end > seq_len:
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

    avg_logits = sum_logits / count.clamp(min=1.0)
    frame_level_preds = torch.argmax(avg_logits, dim=-1).cpu().numpy()

    return frame_level_preds

def predict_and_save(model,test_dataset,output_dir):
    all_preds = []
    os.makedirs(output_dir, exist_ok=True)
    
    for row in tqdm(test_dataset, desc="predicting"):
        v_feat = row["visual_feat"]
        a_feat = row["audio_feat"]
        v_mask = row["v_mask"]
        v_name = row ["video_name"]
        preds = predict_single_video(
            model,
            v_feat,
            a_feat,
            v_mask,
            window_size=64,
            stride=8,
            device=DEVICE,
        )

        preds = medfilt(preds, kernel_size=11)
        preds = preds.astype(np.int64)
        all_preds.extend(preds)
        output_path = os.path.join(output_dir, f"{v_name}.txt")
        with open(output_path, "x")as f:
            for pred in preds:
                f.write(f"{pred}\n")
            

def evaluate(model, test_dataset):
    all_preds = []
    all_labels = []
    
    for row in tqdm(test_dataset, desc="evaluating"):
        v_feat = row["visual_feat"]
        a_feat = row["audio_feat"]
        v_mask = row["v_mask"]
        labels = np.array(row["label"])

        preds = predict_single_video(
            model,
            v_feat,
            a_feat,
            v_mask,
            window_size=64,
            stride=8,
            device=DEVICE,
        )

        preds = medfilt(preds, kernel_size=11)

        preds = preds.astype(np.int64)
        
        valid_idx = labels != -1
        if valid_idx.sum() > 0:
            all_preds.extend(preds[valid_idx])
            all_labels.extend(labels[valid_idx])

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    metrics = {
        f"accuracy": accuracy,
        f"macro_f1": f1,
    }

    print(f"\n[Validation] Accuracy: {accuracy:.4f} | Macro F1: {f1:.4f}")
    print("=" * 50 + "\n")

    return metrics


if __name__ == "__main__":
    #predict_and_save(model=model,test_dataset=val_dataset,output_dir="test_result")
    evaluate(model=model,test_dataset=val_dataset)