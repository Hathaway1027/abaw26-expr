import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_from_disk
from safetensors.torch import load_file

from abaw_model.dual_window_att_model.model import DualWindowAttModel


def ensemble_and_smooth_predict(
    models,
    v_feat_full,
    a_feat_full,
    v_mask_full,
    window_size=64,
    stride=4,
    smooth_kernel_size=27,
    device="cuda",
):
    
    seq_len = v_feat_full.shape[0]
    num_classes = 8
    weights = [0.3, 0.3, 0.4]
    
    v_feat_full = torch.FloatTensor(v_feat_full).to(device)
    a_feat_full = torch.FloatTensor(a_feat_full).to(device)
    v_mask_full = torch.LongTensor(v_mask_full).to(device)

    avg_ensemble_logits = torch.zeros((seq_len, num_classes), device=device)
   
    for i, model in enumerate(models):
        model.eval()
        sum_logits = torch.zeros((seq_len, num_classes), device=device)
        count = torch.zeros((seq_len, 1), device=device)

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

        avg_ensemble_logits += (sum_logits / count.clamp(min=1.0)) * weights[i]

    if smooth_kernel_size > 1:
        pad = smooth_kernel_size // 2
        logits_reshaped = avg_ensemble_logits.unsqueeze(0).transpose(1, 2)
        padded_logits = F.pad(logits_reshaped, (pad, pad), mode="replicate")
        smoothed_logits = F.avg_pool1d(padded_logits, smooth_kernel_size, stride=1)
        avg_ensemble_logits = smoothed_logits.transpose(1, 2).squeeze(0)

    frame_level_preds = torch.argmax(avg_ensemble_logits, dim=-1).cpu().numpy()
    return frame_level_preds


def main():
    device = "cuda"

    TEST_DATA_PATH = "test/final_features/multimodal"
    OUTPUT_TXT_DIR = "test/final_submission_EX"
    os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)

    ckpt_folders = [
        "final_models/*tra0.77_val0.414",
        "final_models/*tra0.83_val0.42",
        "final_models/tra0.85_val0.44",
    ]

    models = []

    for ckpt_path in ckpt_folders:

        model = DualWindowAttModel(d_model=128, num_layers=1).to(device)

        weight_path = os.path.join(ckpt_path, "model.safetensors")

        model.load_state_dict(load_file(weight_path))
        models.append(model)


    test_dataset = load_from_disk(TEST_DATA_PATH).with_format("numpy")


    for row in tqdm(test_dataset, desc="Generating Submission"):

        video_name = row["video_name"]
        v_feat = row["visual_feat"]
        a_feat = row["audio_feat"]
        v_mask = row["v_mask"]

        preds = ensemble_and_smooth_predict(
            models=models,
            v_feat_full=v_feat,
            a_feat_full=a_feat,
            v_mask_full=v_mask,
            window_size=64,
            stride=4,
            smooth_kernel_size=27,  
            device=device,
        )

        txt_path = os.path.join(OUTPUT_TXT_DIR, f"{video_name}.txt")

        with open(txt_path, "w") as f:

            for p in preds:
                f.write(f"{p}\n")


if __name__ == "__main__":

    main()
