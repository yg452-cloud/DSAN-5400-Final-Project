import pandas as pd
from typing import Dict, List, Tuple, Any


"""
Emotion aggregation utilities for the Echo to Empathy project.

This module does NOT train any neural model. Instead, it assumes that
each comment already has fine-grained GoEmotions labels (0/1 columns)
and aggregates them into:
    (1) a macro emotion label (one of a small set of categories)
    (2) a continuous valence score in [-1.0, 1.0]

It is designed to work directly with the `parent_child_pairs.parquet`
dataset, where emotion columns have suffixes like `_child` or `_parent`.
"""

# ---------------------------------------------------------------------------
# 1. GoEmotions base labels (no suffix)
# ---------------------------------------------------------------------------

GOEMOTION_BASE: List[str] = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
    "example_very_unclear",
]


# ---------------------------------------------------------------------------
# 2. Map fine-grained labels into macro emotion categories
#    (these are the 5–6 high-level groups used in downstream analysis)
# ---------------------------------------------------------------------------

EMOTION_TO_MACRO: Dict[str, str] = {
    # Joy / positive-affect group
    "admiration": "joy",
    "amusement": "joy",
    "approval": "joy",
    "caring": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "joy": "joy",
    "love": "joy",
    "optimism": "joy",
    "pride": "joy",
    "relief": "joy",

    # Anger / frustration group
    "anger": "anger",
    "annoyance": "anger",
    "disappointment": "anger",
    "disapproval": "anger",

    # Sadness / loss group
    "grief": "sadness",
    "remorse": "sadness",
    "sadness": "sadness",

    # Fear / anxiety group
    "fear": "fear",
    "nervousness": "fear",

    # Other negative group
    "confusion": "other_negative",
    "embarrassment": "other_negative",
    "disgust": "other_negative",

    # Neutral / informational or ambiguous group
    "neutral": "neutral",
    "realization": "neutral",
    "curiosity": "neutral",
    "desire": "neutral",
    "example_very_unclear": "neutral",
}


# ---------------------------------------------------------------------------
# 3. Assign each base emotion a valence score in [-1.0, 1.0]
# ---------------------------------------------------------------------------

EMOTION_VALENCE: Dict[str, float] = {
    # Strong positive emotions
    "admiration": 0.7,
    "amusement": 0.8,
    "approval": 0.6,
    "caring": 0.6,
    "excitement": 0.8,
    "gratitude": 0.9,
    "joy": 1.0,
    "love": 0.9,
    "optimism": 0.8,
    "pride": 0.7,
    "relief": 0.6,

    # Negative affect — anger-related emotions
    "anger": -0.9,
    "annoyance": -0.6,
    "disappointment": -0.7,
    "disapproval": -0.6,

    # Negative affect — sadness-related emotions
    "grief": -0.9,
    "remorse": -0.7,
    "sadness": -0.8,

    # Negative affect — fear-related emotions
    "fear": -0.8,
    "nervousness": -0.6,

    # Other negative emotions
    "confusion": -0.3,
    "embarrassment": -0.5,
    "disgust": -0.8,

    # Neutral / informational emotions
    "neutral": 0.0,
    "realization": 0.0,
    "curiosity": 0.1,
    "desire": 0.2,
    "example_very_unclear": 0.0,
}


# ---------------------------------------------------------------------------
# 4. EmotionAggregator class
# ---------------------------------------------------------------------------

class EmotionAggregator:
    """
    Aggregates fine-grained GoEmotions labels (0/1 indicator columns)
    into a macro label and valence score for each comment.

    Parameters
    ----------
    role : str
        Either "child" or "parent". This controls which suffix is used:
        e.g. "anger_child" vs "anger_parent", and which ID column is used
        as the comment identifier.
    """

    def __init__(self, role: str = "child") -> None:
        if role not in ("child", "parent"):
            raise ValueError("role must be 'child' or 'parent'")

        self.role = role
        self.suffix = "_child" if role == "child" else "_parent"
        self.id_column = "id_child" if role == "child" else "id_parent"

    # ------------------------------------------------------------------
    # Helper for a single row
    # ------------------------------------------------------------------
    def aggregate_row_emotion(self, row: pd.Series) -> Dict[str, Any]:
        """
        Given a single row from the parent_child_pairs DataFrame,
        determine the macro emotion label and valence for that comment.

        The function looks for columns of the form "<emotion><suffix>",
        e.g. "joy_child" or "anger_parent", depending on the configured
        role/suffix.
        """
        active_emotions = []

        # Identify which base emotions are active (indicator == 1)
        for emo in GOEMOTION_BASE:
            col_name = f"{emo}{self.suffix}"
            if col_name in row.index:
                try:
                    value = row[col_name]
                except KeyError:
                    continue
                if value == 1:
                    active_emotions.append(emo)

        # If no emotion is active, default to neutral
        if len(active_emotions) == 0:
            return {"macro_label": "neutral", "valence": 0.0}

        # Count macro emotion groups and collect valence scores
        macro_counts: Dict[str, int] = {}
        valence_scores: List[float] = []

        for emo in active_emotions:
            macro = EMOTION_TO_MACRO.get(emo, "neutral")
            macro_counts[macro] = macro_counts.get(macro, 0) + 1

            # Fallback to 0.0 if a label is missing from EMOTION_VALENCE
            val = EMOTION_VALENCE.get(emo, 0.0)
            valence_scores.append(val)

        # Dominant macro emotion = the one with the highest count
        macro_label = max(macro_counts, key=macro_counts.get)

        # Mean valence across all active emotions
        valence = float(sum(valence_scores) / len(valence_scores))

        return {"macro_label": macro_label, "valence": valence}

    # ------------------------------------------------------------------
    # Process an entire DataFrame
    # ------------------------------------------------------------------
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply emotion aggregation to an entire DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The parent_child_pairs-like DataFrame containing
            columns such as "joy_child", "anger_child", etc.
            and an ID column (id_child or id_parent).

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns:
                - comment_id
                - macro_label
                - valence
        """
        records = []

        for _, row in df.iterrows():
            agg = self.aggregate_row_emotion(row)

            comment_id = row.get(self.id_column, None)
            records.append(
                {
                    "comment_id": comment_id,
                    "macro_label": agg["macro_label"],
                    "valence": agg["valence"],
                }
            )

        results = pd.DataFrame.from_records(records)

        return results


    # ------------------------------------------------------------------
    # Convenience function for reading from a parquet file
    # ------------------------------------------------------------------
    def process_parquet(self, parquet_path: str) -> pd.DataFrame:
        """
        Load a parquet file and aggregate emotions for all rows.

        This is a convenience wrapper used by scripts.
        """
        df = pd.read_parquet(parquet_path)
        return self.process_dataframe(df)
