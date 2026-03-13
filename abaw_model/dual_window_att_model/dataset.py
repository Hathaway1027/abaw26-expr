import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
import numpy as np
import torch.nn as nn


class AffWild2WindowDataset(Dataset):
    def __init__(self, hf_dataset_path, window_size=64, stride=16, num_classes=8):
        self.window_size = window_size
        self.stride = stride
        self.num_classes = num_classes

        print("Loading...")
        self.ds = load_from_disk(hf_dataset_path).with_format("numpy")

        self.windows = []

        # 初始化计数器，用于统计 0-7 每一类的总样本数
        self.class_counts = np.zeros(num_classes, dtype=np.int64)

        print("Pre-scanning valid windows and calculating statistics...")

        # 预计算合法的滑动窗口索引
        for idx, row in enumerate(self.ds):
            labels = row["label"]
            seq_len = row["seq_length"]

            # 遍历该视频的所有窗口
            for start in range(0, seq_len - window_size + 1, stride):
                end = start + window_size
                window_labels = labels[start:end]

                # 如果窗口内-1的帧超过 1/4，直接丢弃
                if np.sum(window_labels == -1) > (window_size / 4):
                    continue

                self.windows.append((idx, start))

                # 提取有效标签
                valid_labels = window_labels[window_labels != -1]

                #  统计当前窗口内各类别数量
                if len(valid_labels) > 0:
                    counts = np.bincount(
                        valid_labels.astype(np.int64), minlength=self.num_classes
                    )

                    self.class_counts += counts[: self.num_classes]

        print(f"Total valid windows for training: {len(self.windows)}")
        print(f"Class counts: {self.class_counts}")

    def get_class_weights(self, method="effective_num", beta=0.9999):
        
        counts = self.class_counts
        total = np.sum(counts)

        if total == 0:
            return None

        if method == "inverse":
            weights = total / (counts * len(counts) + 1e-6)

        elif method == "effective_num":
            # CVPR 2019: Class-Balanced Loss Based on Effective Number of Samples
            effective_num = 1.0 - np.power(beta, counts)
            weights = (1.0 - beta) / (effective_num + 1e-6)

        else:
            weights = np.ones_like(counts)

        weights = weights / np.mean(weights)

        return torch.FloatTensor(weights)

    def __getitem__(self, idx):
        ds_idx, start = self.windows[idx]
        end = start + self.window_size

        row = self.ds[ds_idx]
        v_feat = row["visual_feat"][start:end]
        a_feat = row["audio_feat"][start:end]
        v_mask = row["v_mask"][start:end]
        label = row["label"][start:end]

        return {
            "v_in": torch.FloatTensor(v_feat),  # (W, V_dim)
            "a_in": torch.FloatTensor(a_feat),  # (W, A_dim)
            "v_mask": torch.LongTensor(v_mask),  # (W,)  0代表长缺失(全0), 1代表有效
            "labels": torch.LongTensor(label),  # (W,)
        }

    def __len__(self):
        return len(self.windows)
