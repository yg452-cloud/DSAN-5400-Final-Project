import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt


def load_data():
    return pd.read_parquet("data/contagion_ready.parquet")


def compute_propagation_strength(df):
    """
    Computes emotion-specific contagion strength:
    P(child = same emotion | parent emotion) - baseline(child emotion)
    """
    parent = df["emotion_parent"]
    child = df["emotion_child"]

    # All unique macro emotions
    emotions = sorted(df["emotion_parent"].unique())

    # Transition probability matrix (parent to child)
    transition_probs = pd.crosstab(parent, child, normalize="index")

    # Baseline frequency of each child emotion
    baseline = child.value_counts(normalize=True).to_dict()

    results = {}

    for em in emotions:
        if em in transition_probs.index and em in transition_probs.columns:
            p_same = transition_probs.loc[em, em]
            p_base = baseline[em]
            contagion_strength = p_same - p_base

            results[em] = {
                "p_same": float(p_same),
                "baseline": float(p_base),
                "contagion_strength": float(contagion_strength)
            }

    # Convert to DataFrame for ranking
    df_results = pd.DataFrame(results).T
    df_results = df_results.sort_values("contagion_strength", ascending=False)

    return df_results


def plot_propagation(df_results):
    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.bar(df_results.index, df_results["contagion_strength"], color="#5DADE2")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Emotion Propagation Strength (Above Baseline)")
    plt.ylabel("Contagion Strength")
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path = "figures/emotion_propagation_strength.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved figure: {out_path}")


def save_results(df_results):
    os.makedirs("results", exist_ok=True)

    df_results.to_csv("results/emotion_propagation.csv")
    df_results.to_json("results/emotion_propagation.json", indent=4)
    print("Saved propagation strength results.")


if __name__ == "__main__":
    df = load_data()
    print("Loaded data:", df.shape)

    df_results = compute_propagation_strength(df)
    print("\nEmotion Propagation Strength Ranking:")
    print(df_results)

    save_results(df_results)
    plot_propagation(df_results)
