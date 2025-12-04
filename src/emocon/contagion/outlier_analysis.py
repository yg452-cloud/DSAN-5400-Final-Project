import pandas as pd
import numpy as np
import json
import os


def load_data():
    return pd.read_parquet("data/contagion_ready.parquet")


def compute_parent_propagation(df):
    df = df.copy()
    df["match"] = (df["emotion_parent"] == df["emotion_child"]).astype(int)

    grouped = (
        df.groupby("parent_id")
        .agg(
            parent_emotion=("emotion_parent", "first"),
            n_children=("child_id", "count"),
            n_matches=("match", "sum"),
        )
        .reset_index()
    )

    grouped["match_rate"] = grouped["n_matches"] / grouped["n_children"]

    return grouped


def identify_outliers(grouped):
    """
    Outlier definitions:
    - Perfect propagators: match_rate == 1.0 and n_children >= 2
    - Strong propagators: match_rate >= 0.75 and n_children >= 3
    - Heavy-tail cases: top 1% match_rate with n_children >= 2
    """

    perfect = grouped[(grouped["match_rate"] == 1.0) & (grouped["n_children"] >= 2)]

    strong = grouped[
        (grouped["match_rate"] >= 0.75)
        & (grouped["match_rate"] < 1.0)
        & (grouped["n_children"] >= 3)
    ]

    threshold = grouped["match_rate"].quantile(0.99)
    heavy_tail = grouped[
        (grouped["match_rate"] >= threshold)
        & (grouped["n_children"] >= 2)
    ]

    return {
        "perfect_propagators": perfect.to_dict(orient="records"),
        "strong_propagators": strong.to_dict(orient="records"),
        "heavy_tail_cases": heavy_tail.to_dict(orient="records"),
        "summary_counts": {
            "n_perfect": len(perfect),
            "n_strong": len(strong),
            "n_heavy_tail": len(heavy_tail),
        },
    }


def save_results(results):
    os.makedirs("results", exist_ok=True)
    out_path = "results/outlier_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved outlier analysis to: {out_path}")


if __name__ == "__main__":
    df = load_data()

    grouped = compute_parent_propagation(df)

    outliers = identify_outliers(grouped)

    print("\n=== Outlier Summary ===")
    print(outliers["summary_counts"])

    save_results(outliers)
