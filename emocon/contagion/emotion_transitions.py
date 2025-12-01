import pandas as pd
import numpy as np
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt


def load_data():
    return pd.read_parquet("data/contagion_ready.parquet")


def build_transition_matrix(df):
    """
    Create parent→child emotion transition counts and probabilities.
    """
    # Extract the two categorical emotion columns
    parent_em = df["emotion_parent"]
    child_em = df["emotion_child"]

    # Count transitions
    transition_counts = pd.crosstab(parent_em, child_em)

    # Probability matrix = row-normalized counts
    transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)

    return transition_counts, transition_probs


def save_results(counts, probs):
    os.makedirs("emocon/results", exist_ok=True)

    counts.to_csv("emocon/results/emotion_transition_counts.csv")
    probs.to_csv("emocon/results/emotion_transition_probs.csv")

    # Also save JSON 
    counts.to_json("emocon/results/emotion_transition_counts.json", indent=4)
    probs.to_json("emocon/results/emotion_transition_probs.json", indent=4)


def plot_heatmap(probs):
    os.makedirs("emocon/figures", exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(probs, annot=False, cmap="mako", linewidths=0.5)
    plt.title("Emotion Transition Probability Heatmap")
    plt.xlabel("Child Emotion")
    plt.ylabel("Parent Emotion")

    output_path = "emocon/figures/emotion_transition_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved heatmap: {output_path}")


if __name__ == "__main__":
    df = load_data()
    print("Loaded contagion_ready.parquet:", df.shape)

    counts, probs = build_transition_matrix(df)

    print("\nTransition Matrix (Counts):")
    print(counts.head())

    print("\nTransition Matrix (Probabilities):")
    print(probs.head())

    save_results(counts, probs)
    plot_heatmap(probs)

    print("\nSaved transition matrices + heatmap.")
