import os
import torch
import numpy as np
from PIL import Image
import pickle
import torch.nn as nn
from transformers import BeitModel, BeitImageProcessor, ViTModel, ViTImageProcessor,ResNetModel
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from torchvision.models import efficientnet_v2_m


MODEL_PATH = "models/visual-extract/best_beit_affwild2"
BASE_DIR = "abaw_dataset/cropped"
LABEL_DIR = "abaw_dataset/ABAW_Annotations/EXPR_Recognition_Challenge"
SUB_PATH = ["train", "val"]
OUTPUT_DIR = "abaw_dataset/features/visual"

processor = BeitImageProcessor.from_pretrained(
    "microsoft/beit-large-patch16-224-pt22k"
)
model = BeitModel.from_pretrained(MODEL_PATH)
model.name = "beit-large"
model.eval()

output_path = os.path.join(OUTPUT_DIR, model.name)
if not os.path.exists(output_path):
    os.makedirs(output_path)


class VideoFrameDataset(Dataset):
    def __init__(self, data_dir, processor):
        self.data_dir = data_dir
        self.processor = processor
        self.samples = []

        folders = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, f))
            ]
        )

        for sub_dir_name in folders:
            frames_dir = os.path.join(data_dir, sub_dir_name)
            img_list = sorted(
                [img for img in os.listdir(frames_dir) if img.lower().endswith(".jpg")]
            )

            for img_name in img_list:
                relative_path = os.path.join(sub_dir_name, img_name)
                full_path = os.path.join(frames_dir, img_name)
                self.samples.append((relative_path, full_path))

        print(f"共找到 {len(self.samples)} 张图片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, full_path = self.samples[idx]

        img = Image.open(full_path).convert("RGB")
        if img.size[0] == 0:
            raise ValueError("Empty Image")
        return {
            "pixel_values": self.processor(images=img, return_tensors="pt")[
                "pixel_values"
            ].squeeze(0)
        }


class FeatureExtractorWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values)[1]  # Pooler output


def get_total_frames_from_txt(video_folder_name, label_dir):
    
    txt_name = video_folder_name + ".txt"
    txt_path = os.path.join(label_dir, txt_name)

    with open(txt_path, "r") as f:
        count = sum(1 for _ in f)
    return count-1


def extract_and_save(sub_path):
    print(f"--- 正在处理子集: {sub_path} ---")
    data_dir = os.path.join(BASE_DIR, sub_path)
    current_txt_dir = os.path.join(LABEL_DIR,sub_path)
    output_pkl = output_path + f"/{sub_path}_videoname2feature_visual.pkl"

    dataset = VideoFrameDataset(data_dir, processor=processor)
    if len(dataset) == 0:
        return

    wrapper_model = FeatureExtractorWrapper(model)

    args = TrainingArguments(
        output_dir="./tmp_output",
        per_device_eval_batch_size=128,
        dataloader_num_workers=8,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(model=wrapper_model, args=args)
 
    output = trainer.predict(dataset)
    global_features = output.predictions  # (N, D)
    feature_dim = global_features.shape[1]

    temp_data = {}

    for i, (rel_path, full_path) in enumerate(dataset.samples):
        
        folder_name, file_name = os.path.split(rel_path)

        frame_idx = int(os.path.splitext(file_name)[0]) - 1

        if folder_name not in temp_data:
            temp_data[folder_name] = {}

        temp_data[folder_name][frame_idx] = global_features[i]

    final_data = {}

    for folder_name, frames_dict in temp_data.items():

        seq_len = get_total_frames_from_txt(folder_name, current_txt_dir)
        dense_features = np.zeros((seq_len, feature_dim), dtype=np.float32)
        mask = np.zeros((seq_len,), dtype=np.int8)

  
        for idx, feat in frames_dict.items():
            if idx < seq_len:
                dense_features[idx] = feat
                mask[idx] = 1
            else:
                print(f"[错误] 索引越界 {folder_name}: Frame {idx} > Len {seq_len}")

        final_data[folder_name] = {
            "features": dense_features,
            "mask": mask,
        }

    print(f"正在保存到 {output_pkl} ...")
    with open(output_pkl, "wb") as f:
        pickle.dump(final_data, f)

    print(f"处理完成: {len(final_data)} 个视频序列。")


if __name__ == "__main__":
    for sub_path in SUB_PATH:
        extract_and_save(sub_path)
