import pandas as pd
from emocon.models.emotion_model import EmotionAggregator

# Load the parent-child dataset
df = pd.read_parquet("data/parent_child_pairs.parquet")

print("Full df shape:", df.shape)
df_sample = df.head(5)
print("Sample df shape:", df_sample.shape)

# Use the aggregator for child comments
aggregator = EmotionAggregator(role="child")

result = aggregator.process_dataframe(df_sample)

print("Result DataFrame:")
print(result)

