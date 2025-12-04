import pandas as pd
import scipy.stats as stats
import json
import os


def load_clean_data():
    return pd.read_parquet("data/contagion_ready.parquet")


def compute_valence_contagion(df):
    """
    Computes Pearson and Spearman correlations between
    parent valence and child valence.
    """

    parent = df["valence_parent"]
    child = df["valence_child"]

    pearson_r, pearson_p = stats.pearsonr(parent, child)
    spearman_r, spearman_p = stats.spearmanr(parent, child)

    result = {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }

    return result


if __name__ == "__main__":
    df = load_clean_data()
    print("Loaded contagion_ready.parquet with shape:", df.shape)

    results = compute_valence_contagion(df)

    print("\nValence Contagion Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # Save results
    os.makedirs("results", exist_ok=True)

    output_path = "results/contagion_stats.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved: {output_path}")
