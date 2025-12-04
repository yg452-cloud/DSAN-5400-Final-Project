import pandas as pd
import numpy as np
import json
import os
from scipy.stats import chi2_contingency, fisher_exact, norm


def load_data():
    return pd.read_parquet("data/contagion_ready.parquet")


def compute_transition_matrix(df):
    ct = pd.crosstab(df["emotion_parent"], df["emotion_child"])
    return ct


def chi_square_test(matrix):
    chi2, p, dof, expected = chi2_contingency(matrix)
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "degrees_of_freedom": int(dof)
    }


def proportion_z_test(success1, n1, success2, n2):
    """
    Two-proportion z-test.
    """
    p1 = success1 / n1
    p2 = success2 / n2
    p_pool = (success1 + success2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

    if se == 0:
        return {"z": None, "p_value": None, "interpretation": "Standard error is zero."}

    z = (p1 - p2) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))

    return {
        "p1": float(p1),
        "p2": float(p2),
        "z": float(z),
        "p_value": float(p_value)
    }


def compute_depth_significance(df):
    df["match"] = (df["emotion_parent"] == df["emotion_child"]).astype(int)

    d1 = df[df["depth_child_original"] == 1]
    d2 = df[df["depth_child_original"] == 2]

    success1 = d1["match"].sum()
    n1 = len(d1)

    success2 = d2["match"].sum()
    n2 = len(d2)

    return proportion_z_test(success1, n1, success2, n2)


def save_results(results):
    os.makedirs("results", exist_ok=True)
    out_path = "results/significance_stats.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved significance stats to: {out_path}")


if __name__ == "__main__":
    df = load_data()

    # 1. Chi-square test on emotion transitions
    matrix = compute_transition_matrix(df)
    chi_results = chi_square_test(matrix)

    # 2. Depth 1 vs 2 decay significance
    decay_sig = compute_depth_significance(df)

    results = {
        "chi_square_transition_test": chi_results,
        "depth_decay_significance": decay_sig,
        "note": "Z-test evaluates whether emotional match decreases significantly with depth."
    }

    print("\nChi-Square Test for Emotional Dependence:")
    print(chi_results)

    print("\nDepth Decay Significance (Two-Proportion Z-Test):")
    print(decay_sig)

    save_results(results)
