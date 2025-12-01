import pandas as pd


def load_and_merge_data():
    """
    Loads all data and merges it into a clean dataframe for
    emotional contagion analysis.

    Output columns:
        parent_id
        child_id
        emotion_parent
        valence_parent
        emotion_child
        valence_child
        depth_parent
        depth_child
        delta_depth
    """

    # ------------------------------------------------------------
    # 1. LOAD ALL REQUIRED FILES
    # ------------------------------------------------------------
    pairs = pd.read_parquet("data/parent_child_pairs.parquet")
    parent_scores = pd.read_parquet("data/emotion_scores_parent.parquet")
    child_scores = pd.read_parquet("data/emotion_scores_child.parquet")
    threads = pd.read_parquet("data/threads_with_replies.parquet")

    # ------------------------------------------------------------
    # 2. CLEAN ID COLUMNS IN PAIRS (STAGE 1 OUTPUT)
    # ------------------------------------------------------------
    # Keep only what Stage 3 needs
    pairs_clean = pairs[["id_parent", "id_child", "depth"]].copy()

    pairs_clean = pairs_clean.rename(columns={
        "id_parent": "parent_id",
        "id_child": "child_id",
        "depth": "depth_child_original"
    })

    # ------------------------------------------------------------
    # 3. PREPARE EMOTION SCORES (STAGE 2 OUTPUT)
    #    Ensure ONE ROW PER COMMENT ID before merging
    # ------------------------------------------------------------
    parent_scores = parent_scores.rename(columns={
        "comment_id": "parent_id",
        "macro_label": "emotion_parent",
        "valence": "valence_parent"
    })

    child_scores = child_scores.rename(columns={
        "comment_id": "child_id",
        "macro_label": "emotion_child",
        "valence": "valence_child"
    })

    # Enforce uniqueness: one row per parent_id / child_id
    parent_scores = parent_scores.drop_duplicates(subset="parent_id")
    child_scores = child_scores.drop_duplicates(subset="child_id")

    # ------------------------------------------------------------
    # 4. MERGE PARENT/CHILD IDS WITH EMOTION SCORES
    # ------------------------------------------------------------
    df = pairs_clean.merge(parent_scores, on="parent_id", how="inner")
    df = df.merge(child_scores, on="child_id", how="inner")

    # ------------------------------------------------------------
    # 5. PREPARE THREAD DEPTH INFO (threads_with_replies.parquet)
    # ------------------------------------------------------------
    # threads_with_replies uses "id" as the comment ID
    thread_depth = threads[["id", "depth"]]

    # ------------------------------------------------------------
    # 6. MERGE PARENT DEPTH
    # ------------------------------------------------------------
    df = df.merge(
        thread_depth.rename(columns={
            "id": "parent_id",
            "depth": "depth_parent"
        }),
        on="parent_id",
        how="left"
    )

    # ------------------------------------------------------------
    # 7. MERGE CHILD DEPTH
    # ------------------------------------------------------------
    df = df.merge(
        thread_depth.rename(columns={
            "id": "child_id",
            "depth": "depth_child"
        }),
        on="child_id",
        how="left"
    )

    # ------------------------------------------------------------
    # 8. COMPUTE DELTA DEPTH
    # ------------------------------------------------------------
    df["delta_depth"] = df["depth_child"] - df["depth_parent"]

    # (Optional) drop the original depth_child from pairs if you don't need it
    # df = df.drop(columns=["depth_child_original"])

    return df


if __name__ == "__main__":
    df = load_and_merge_data()
    print(df.head())
    print("\nShape:", df.shape)

