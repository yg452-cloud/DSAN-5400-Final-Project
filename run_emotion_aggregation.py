import pandas as pd
from emocon.models.emotion_model import EmotionAggregator


def main():
    # ------------------------------------------------------------------
    # 1. Load the full parent-child dataset
    # ------------------------------------------------------------------
    df = pd.read_parquet("data/parent_child_pairs.parquet")
    print("Loaded parent_child_pairs.parquet with shape:", df.shape)

    # ------------------------------------------------------------------
    # 2. Aggregate emotions for CHILD comments
    # ------------------------------------------------------------------
    child_aggregator = EmotionAggregator(role="child")
    child_results = child_aggregator.process_dataframe(df)
    print("Child results shape:", child_results.shape)

    # Save to parquet
    child_output_path = "data/emotion_scores_child.parquet"
    child_results.to_parquet(child_output_path, index=False)
    print(f"Saved child emotion scores to: {child_output_path}")

    # ------------------------------------------------------------------
    # 3. (Optional) Aggregate emotions for PARENT comments as well
    # ------------------------------------------------------------------
    parent_aggregator = EmotionAggregator(role="parent")
    parent_results = parent_aggregator.process_dataframe(df)
    print("Parent results shape:", parent_results.shape)

    parent_output_path = "data/emotion_scores_parent.parquet"
    parent_results.to_parquet(parent_output_path, index=False)
    print(f"Saved parent emotion scores to: {parent_output_path}")


if __name__ == "__main__":
    main()
