import os
import pandas as pd
import numpy as np
from datasets import Dataset, Features, ClassLabel, Image, Value, load_dataset
from PIL import Image as PILImage
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

CONFIG = {
    "train_csv": "affwild2_train.csv",
    "val_csv": "affwild2_val.csv",
  
    "output_train": "Affwild2_ft/train.parquet",
    "output_val": "Affwild2_ft/val.parquet",
}


def process_csv_dataset(csv_path):
    if not os.path.exists(csv_path):
        print(f"警告: 文件不存在 {csv_path}")
        return []

    print(f"正在扫描 {csv_path} ...")
    metadata = []

    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():

        label = int(row["label"])
        img_filename = row["image_path"]

        metadata.append(
            {
                "image_path": img_filename,
                "label": label,
            }
        )

    return metadata


def image_generator(metadata_list):
    
    for item in metadata_list:
        path = item["image_path"]
        try:
            with open(path, "rb") as f:
                img_bytes = f.read()

            yield {
                "image": {"bytes": img_bytes, "path": None},
                "label": item["label"],
            }
        except Exception as e:
            print(f"读取错误: {path}, Error: {e}")
            continue


def main():

    train = process_csv_dataset(CONFIG["train_csv"])
    val = process_csv_dataset(CONFIG["val_csv"])

   
    print(f"训练集总数: {len(train)}")

    print(f"验证集总数: {len(val)}")

    features = Features(
        {
            "image": Image(), 
            "label": ClassLabel(
                num_classes=8,
                names=[
                    "Neutral",
                    "Anger",
                    "Disgust",
                    "Fear",
                    "Happiness",
                    "Sadness",
                    "Surprise",
                    "Other"
                ],
            )
        }
    )

    print(f"正在生成训练集 Parquet: {CONFIG['output_train']} ...")
    ds_train = Dataset.from_generator(
        lambda: image_generator(train),
        features=features,
    )
    ds_train.to_parquet(CONFIG["output_train"])

    print(f"正在生成验证集 Parquet: {CONFIG['output_val']} ...")
    ds_val = Dataset.from_generator(
        lambda: image_generator(val),
        features=features,
    )
    ds_val.to_parquet(CONFIG["output_val"])

    print("全部完成！")

if __name__ == "__main__":
    main()
