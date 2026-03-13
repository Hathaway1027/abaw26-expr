import os
import pandas as pd
import numpy as np
from datasets import Dataset, Features, ClassLabel, Image, Value, load_dataset
from PIL import Image as PILImage
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

CONFIG = {
    # --- RAF-DB ---
    "raf_db": {
        "train_csv": "dataset/raf_db/processed_training.csv",
        "val_csv": "dataset/raf_db/processed_validation.csv",
        "img_root": "dataset/raf_db/image",
    },
    # --- AffectNet ---
    "affectnet": {
        "train_csv": "dataset/AffectNet/processed_training.csv",
        "train_img_root": "dataset/Aligned_AffectNet/training",
        "val_csv": "dataset/AffectNet/processed_validation.csv",
        "val_img_root": "dataset/Aligned_AffectNet/validation",
    },
    # --- FERPlus ---
    "ferplus": {
        "train_csv": "dataset/ferplus/FER2013Train/label.csv",
        "train_img_root": "dataset/ferplus/FER2013Train",
        "val_csv": "dataset/ferplus/FER2013Valid/label.csv",
        "val_img_root": "dataset/ferplus/FER2013Valid",
        "test_csv": "dataset/ferplus/FER2013Test/label.csv",
        "test_img_root": "dataset/ferplus/FER2013Test",
    },
    # --- 输出配置 ---
    "output_train": "full_dataset/rafdb_ferplus_affectnet_train.parquet",
    "output_val": "full_dataset/rafdb_ferplus_affectnet_validation.parquet",
}


#0anger 1disgust 2fear 3happy 4neutral 5sad 6surprise
MAP_RAF = {1: 6, 2: 2, 3: 1, 4: 3, 5: 5, 6: 0, 7: 4}
MAP_AFFECTNET = {0: 4, 1: 3, 2: 5, 3: 6, 4: 2, 5: 1, 6: 0}
MAP_FERPLUS = {0: 4, 1: 3, 2: 6, 3: 5, 4: 0, 5: 1, 6: 2}


def process_csv_dataset(csv_path, img_root, source_name, mapping, is_ferplus=False):
    if not os.path.exists(csv_path):
        print(f"警告: 文件不存在 {csv_path}")
        return []

    print(f"正在扫描 {source_name} - {csv_path} ...")
    metadata = []

    if is_ferplus:
        
        df = pd.read_csv(csv_path, header=None)

        for _, row in df.iterrows():

            img_filename = row.iloc[0].strip()
            votes = row.iloc[2:9].values.astype(int)
            max_idx = np.argmax(votes) 

            if max_idx in mapping:
                target_label = mapping[max_idx]
                full_path = os.path.join(img_root, img_filename)
                
                metadata.append(
                    {
                        "image_path": full_path,
                        "source": source_name,
                        "label": target_label,
                    }
                )

    else:
        # AffectNet/RAF-DB 
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():

            orig = int(row["expression"])
            if orig in mapping:
                target_label = mapping[orig]
                img_filename = row["subDirectory_filePath"]
                full_path = os.path.join(img_root, img_filename)
          
                metadata.append(
                    {
                        "image_path":full_path,
                        "source": source_name,
                        "label": target_label,
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
                "source": item["source"],
            }
        except Exception as e:
            print(f"读取错误: {path}, Error: {e}")
            continue


def main():

    raf_train = process_csv_dataset(
        CONFIG["raf_db"]["train_csv"],
        CONFIG["raf_db"]["img_root"],
        "raf_db",
        MAP_RAF,
        is_ferplus=False,
    )
    raf_val = process_csv_dataset(
        CONFIG["raf_db"]["val_csv"],
        CONFIG["raf_db"]["img_root"],
        "raf_db",
        MAP_RAF,
        is_ferplus=False,
    )

    
    aff_train = process_csv_dataset(
        CONFIG["affectnet"]["train_csv"],
        CONFIG["affectnet"]["train_img_root"],
        "affectnet",
        MAP_AFFECTNET,
        is_ferplus=False,
    )
    aff_val = process_csv_dataset(
        CONFIG["affectnet"]["val_csv"],
        CONFIG["affectnet"]["val_img_root"],
        "affectnet",
        MAP_AFFECTNET,
        is_ferplus=False,
    )

    
    fer_train = process_csv_dataset(
        CONFIG["ferplus"]["train_csv"],
        CONFIG["ferplus"]["train_img_root"],
        "ferplus",
        MAP_FERPLUS,
        is_ferplus=True,
    )
    fer_val = process_csv_dataset(
        CONFIG["ferplus"]["val_csv"],
        CONFIG["ferplus"]["val_img_root"],
        "ferplus",
        MAP_FERPLUS,
        is_ferplus=True,
    )
    fer_test = process_csv_dataset(
        CONFIG["ferplus"]["test_csv"],
        CONFIG["ferplus"]["test_img_root"],
        "ferplus",
        MAP_FERPLUS,
        is_ferplus=True,
    )

    global_train_meta = aff_train + fer_train + raf_train
    print(f"训练集总数: {len(global_train_meta)}")
   
    global_val_meta = aff_val + fer_val + fer_test + raf_val
    print(f"验证集总数: {len(global_val_meta)}")
    
    features = Features(
        {
            "image": Image(), 
            "label": ClassLabel(
                num_classes=7,
                names=[
                    "anger",
                    "disgust",
                    "fear",
                    "happy",
                    "neutral",
                    "sad",
                    "surprise",
                ],
            ),
            "source": Value("string"),
        }
    )

    print(f"正在生成训练集 Parquet: {CONFIG['output_train']} ...")
    ds_train = Dataset.from_generator(
        lambda: image_generator(global_train_meta),
        features=features,
    )
    ds_train.to_parquet(CONFIG["output_train"])

    print(f"正在生成验证集 Parquet: {CONFIG['output_val']} ...")
    ds_val = Dataset.from_generator(
        lambda: image_generator(global_val_meta),
        features=features,
    )
    ds_val.to_parquet(CONFIG["output_val"])

    print("全部完成！")


if __name__ == "__main__":
    main()
