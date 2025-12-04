# Examples

This directory contains example scripts demonstrating how to use the Emocon package.

## Available Examples

### 1. `download_data.py`
Downloads the GoEmotions dataset from HuggingFace.

**Usage:**
```bash
python examples/download_data.py
```

**Equivalent CLI command:**
```bash
emocon download
```

---

### 2. `run_emotion_aggregation.py`
Demonstrates the full emotion aggregation pipeline:
- Loads parent-child pairs
- Aggregates emotions for both parent and child comments
- Saves emotion scores to parquet files

**Usage:**
```bash
python examples/run_emotion_aggregation.py
```

**Equivalent CLI command:**
```bash
emocon aggregate-emotions
```

**Output files:**
- `data/emotion_scores_child.parquet`
- `data/emotion_scores_parent.parquet`

---

### 3. `run_aggregator_test.py`
Quick test of the EmotionAggregator on a small sample.

**Usage:**
```bash
python examples/run_aggregator_test.py
```

**What it does:**
- Loads first 5 rows from parent-child pairs
- Runs emotion aggregation
- Displays results for inspection

---

### 4. `check_emotion_scores.py`
Verifies that emotion score files were created correctly.

**Usage:**
```bash
python examples/check_emotion_scores.py
```

**What it checks:**
- File shapes and structure
- Column names
- Sample data

---

## Using Python API Directly

Instead of running these scripts, you can use the Emocon package directly in your own code:

```python
from emocon import EmotionAggregator, RedditDataLoader
import pandas as pd

# Download data
RedditDataLoader.download_from_huggingface()

# Load and aggregate emotions
df = pd.read_parquet("data/parent_child_pairs.parquet")
aggregator = EmotionAggregator(role="child")
results = aggregator.process_dataframe(df)

print(results.head())
```

---

## CLI Alternative

All these examples can be replaced by CLI commands:

```bash
# Full pipeline
emocon analyze

# Individual stages
emocon download
emocon preprocess
emocon aggregate-emotions
emocon prepare-contagion
```

See `emocon --help` for more options.
