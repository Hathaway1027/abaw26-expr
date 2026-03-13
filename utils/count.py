import pandas as pd
from datasets import load_from_disk
import numpy as np

ds = load_from_disk(
    "abaw_dataset/features/multimodal_feature/beit-large/train"
)

all_labels = []
for sample in ds:
    all_labels.extend(sample["label"])

possible_labels = [-1, 0, 1, 2, 3, 4, 5, 6, 7]
counts = pd.Series(all_labels).value_counts()
result = pd.DataFrame({"label": possible_labels})
result["count"] = result["label"].map(counts).fillna(0).astype(int)
result = result.sort_values("label").reset_index(drop=True)

result.to_csv("trainds_label_distribution.csv", index=False)
