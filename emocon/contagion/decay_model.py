import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt


def load_data():
    return pd.read_parquet("data/contagion_ready.parquet")


def compute_depth_decay(df):
    """
    Compute P(match | child absolute depth) where
    match = 1 if parent and child share the same macro emotion.
    """
    df = df.copy()
    df["emotion_match"] = (df["emotion_parent"] == df["emotion_child"]).astype(int)

    # Use the original child depth from parent_child_pairs
    if "depth_child_original" not in df.columns:
        raise KeyError("depth_child_original not found in dataframe.")

    decay_df = (
        df.groupby("depth_child_original")["emotion_match"]
        .mean()
        .reset_index()
        .rename(columns={"emotion_match": "p_match",
                         "depth_child_original": "depth"})
    )

    return decay_df


def simple_slope(decay_df):
    """
    With only a couple of depth levels, we compute a simple slope:
        slope = (p_match(depth_max) - p_match(depth_min)) / (depth_max - depth_min)
    No formal p-values here – data is too limited.
    """
    if len(decay_df) < 2:
        return {"note": "Only one depth level available; cannot compute slope."}

    decay_df = decay_df.sort_values("depth")
    d1, d2 = decay_df["depth"].iloc[0], decay_df["depth"].iloc[1]
    p1, p2 = decay_df["p_match"].iloc[0], decay_df["p_match"].iloc[1]

    slope = (p2 - p1) / (d2 - d1)
    return {
        "depth_min": int(d1),
        "depth_max": int(d2),
        "p_match_min": float(p1),
        "p_match_max": float(p2),
        "slope": float(slope),
        "interpretation": "Negative slope indicates decay (less matching at deeper replies)."
    }


def plot_decay(decay_df):
    os.makedirs("emocon/figures", exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.scatter(decay_df["depth"], decay_df["p_match"], label="Observed", s=60)
    plt.plot(decay_df["depth"], decay_df["p_match"], linestyle="--")

    plt.title("Emotion Match vs. Child Depth")
    plt.xlabel("Child Depth (absolute)")
    plt.ylabel("P(parent & child share emotion)")
    plt.xticks(sorted(decay_df["depth"].unique()))
    plt.tight_layout()

    out_path = "emocon/figures/decay_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved decay plot: {out_path}")


def save_stats(slope_stats, decay_df):
    os.makedirs("emocon/results", exist_ok=True)

    out = {
        "depth_decay_points": decay_df.to_dict(orient="records"),
        "slope_summary": slope_stats,
        "note": "Decay modeled over child absolute depth levels; dataset has limited depth (1–2)."
    }

    save_path = "emocon/results/decay_stats.json"
    with open(save_path, "w") as f:
        json.dump(out, f, indent=4)

    print(f"Saved stats to: {save_path}")


if __name__ == "__main__":
    df = load_data()

    decay_df = compute_depth_decay(df)
    print("Decay Points (by child depth):")
    print(decay_df)

    slope_stats = simple_slope(decay_df)
    print("\nDecay Slope Summary:")
    print(slope_stats)

    plot_decay(decay_df)
    save_stats(slope_stats, decay_df)
