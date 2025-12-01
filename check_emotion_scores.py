import pandas as pd

child = pd.read_parquet("data/emotion_scores_child.parquet")
parent = pd.read_parquet("data/emotion_scores_parent.parquet")

print("Child scores shape:", child.shape)
print(child.head())

print("\nParent scores shape:", parent.shape)
print(parent.head())
