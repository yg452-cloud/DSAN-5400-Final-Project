# Part 1: Data Acquisition & Thread Graph Construction
**Team Members:** Jiaheng Cao, Matthew Hakim, Yanmin Gui, Ke Tian  
**Course:** DSAN 5400 - Natural Language Processing  
**Project:** Echo to Empathy - Detecting Emotional Contagion in Online Discussions

---

## Overview

This module handles data acquisition, preprocessing, and thread graph construction for the emotional contagion analysis project. It processes Reddit comment data from the GoEmotions dataset and extracts parent-child comment pairs to enable downstream emotion propagation studies.

### Deliverables

- 2,530 parent-child comment pairs for emotion propagation analysis
- 185 conversation threads with reply structures
- Complete data processing pipeline with logging and reproducibility
- Clean, modular Python code following best practices

---

## Project Structure
```
data/
├── goemotions_local.csv              # GoEmotions dataset (download first)
│
├── loader.py                         # Data loading module
├── text_cleaner.py                   # Text preprocessing module
├── thread_builder.py                 # Thread graph construction module
├── logging_config.py                 # Logging configuration
│
├── download_data.py                  # Data download script
├── run_full_pipeline.py              # Main pipeline execution script
│
├── parent_child_pairs.parquet        # OUTPUT: Parent-child comment pairs
└── threads_with_replies.parquet      # OUTPUT: Filtered conversation threads

../logs/
└── data_acquisition.log              # Execution logs for reproducibility
```

---

## Quick Start

### Prerequisites

Required Python packages:
```bash
pip install pandas networkx pyarrow
```

Python version: 3.8 or higher

### Installation

**Step 1: Download Data**

Option A - Using download script:
```bash
cd data
python download_data.py
```

Option B - Using loader module:
```python
from loader import RedditDataLoader
RedditDataLoader.download_from_huggingface()
```

Option C - Manual download:
1. Visit https://huggingface.co/datasets/mrm8488/goemotions
2. Download `goemotions.csv`
3. Save as `data/goemotions_local.csv`

**Step 2: Run Pipeline**
```bash
cd data
python run_full_pipeline.py
```

Expected output:
```
Total comments: 211,219
Threads with replies: 185
Parent-child pairs: 2,530

Files saved:
- threads_with_replies.parquet
- parent_child_pairs.parquet

Logs saved to: ../logs/data_acquisition.log
```

Processing time: approximately 1-2 minutes

---

# Part 1: Emotion Aggregation Module

**Author:** Jiaheng Cao  
**Component:** Emotion Aggregation Layer (Pipeline Stage 2)

## 1. Overview

This module transforms Reddit comments’ fine-grained GoEmotions annotations into:

1. **Macro-level emotion categories**  
   (e.g., *joy, anger, sadness, fear, neutral/other*)
2. **A continuous valence score** in the range `[-1.0, 1.0]`

These standardized emotional signals will be used in the next stage of the project to quantify emotional contagion dynamics between parent and child comments in Reddit threads.

> **Important:** This module does **not** train models or classify text.  
> It **aggregates** already-present GoEmotions labels stored in the dataset.

---

## 2. Data Input Requirements

This component operates on the existing dataset:

'data/parent_child_pairs.parquet'


This file must contain the following structural elements:

| Field Pattern | Description |
|---------------|-------------|
| `<emotion>_child` | GoEmotion indicators for child comments (0/1) |
| `<emotion>_parent` | GoEmotion indicators for parent comments |
| `id_child`, `id_parent` | Unique comment identifiers |

Total rows in our dataset: **2530 comments**  
Total fine-grained GoEmotions labels: **28**

---

## 3. Core Output Artifacts

Running this module generates **two** parquet files:

'data/emotion_scores_child.parquet'
'data/emotion_scores_parent.parquet'


Each file contains:

| Column | Example | Meaning |
|--------|--------|---------|
| `comment_id` | `ed1gxcf` | Reddit comment identifier |
| `macro_label` | `joy` / `anger` / `neutral` | Aggregated dominant emotion |
| `valence` | `0.75`, `-0.8`, `0.2` | Continuous emotional intensity |

These files are the **direct inputs** for Stage 3 of the project:
> Emotional propagation and contagion analysis

---

## 4. Code Structure

emocon/
└── models/
└── emotion_model.py # Main aggregation model

run_aggregator_test.py # Quick validation script (sample only)

run_emotion_aggregation.py # Full pipeline execution

check_emotion_scores.py # Validate final parquet outputs


---

## 5. How to Reproduce Results

### Step 1 — Run sample test (optional sanity check)

```bash
python run_aggregator_test.py
```

**Expected Output (example):**
```bahs
Full df shape: (2530, 71)
Sample df shape: (5, 71)
Result DataFrame:
  comment_id macro_label  valence
0 ed1gxcf    joy         0.75
1 ed1gxcf    joy         0.75
2 ed1gxcf    joy         0.75
3 ed1gxcf    neutral     0.20
4 ed1gxcf    neutral     0.20
```

This confirms the module correctly:

- Reads the parquet file
- Selects child emotion columns
- Maps emotions → macro category
- Computes valence

### Step 2 - Run full emotion aggregation pipeline

```bash
python run_emotion_aggregation.py
```

**Expected terminal output:**

```bash
Loaded parent_child_pairs.parquet with shape: (2530, 71)
Child results shape: (2530, 3)
Saved child emotion scores to: data/emotion_scores_child.parquet
Parent results shape: (2530, 3)
Saved parent emotion scores to: data/emotion_scores_parent.parquet
```

### Step 3 - Verify output files

```bash
python check_emotion_scores.py
```

**Expected terminal output**

```bash
Child scores shape: (2530, 3)
Parent scores shape: (2530, 3)
```

# Part 2: Emotion Aggregation Module

**Author:** Jiaheng Cao  
**Project:** DSAN-5400 Final Project — *Echo to Empathy*  
**Component:** Emotion Aggregation Layer (Pipeline Stage 2)

---

## 1. Overview

This module transforms Reddit comments’ fine-grained GoEmotions annotations into:

1. **Macro-level emotion categories**  
   (e.g., *joy, anger, sadness, fear, neutral/other*)
2. **A continuous valence score** in the range `[-1.0, 1.0]`

These standardized emotional signals will be used in the next stage of the project to quantify emotional contagion dynamics between parent and child comments in Reddit threads.

> **Important:** This module does **not** train models or classify text.  
> It **aggregates** already-present GoEmotions labels stored in the dataset.

---

## 2. Data Input Requirements

This component operates on the existing dataset:

data/parent_child_pairs.parquet


This file must contain the following structural elements:

| Field Pattern | Description |
|---------------|-------------|
| `<emotion>_child` | GoEmotion indicators for child comments (0/1) |
| `<emotion>_parent` | GoEmotion indicators for parent comments |
| `id_child`, `id_parent` | Unique comment identifiers |

Total rows in our dataset: **2530 comments**  
Total fine-grained GoEmotions labels: **28**

---

## 3. Core Output Artifacts

Running this module generates **two** parquet files:

data/emotion_scores_child.parquet

data/emotion_scores_parent.parquet


Each file contains:

| Column | Example | Meaning |
|--------|--------|---------|
| `comment_id` | `ed1gxcf` | Reddit comment identifier |
| `macro_label` | `joy` / `anger` / `neutral` | Aggregated dominant emotion |
| `valence` | `0.75`, `-0.8`, `0.2` | Continuous emotional intensity |

These files are the **direct inputs** for Stage 3 of the project:
> Emotional propagation and contagion analysis

---

## 4. Code Structure

emocon/  
└── models/  
└── emotion_model.py # Main aggregation model

run_aggregator_test.py # Quick validation script (sample only)

run_emotion_aggregation.py # Full pipeline execution

check_emotion_scores.py # Validate final parquet outputs

---

## 5. How to Reproduce Results

### Step 1 — Run sample test (optional sanity check)

```bash
python run_aggregator_test.py
```

**Expected terminal output:**

```bash
Full df shape: (2530, 71)
Sample df shape: (5, 71)
Result DataFrame:
  comment_id macro_label  valence
0 ed1gxcf    joy         0.75
1 ed1gxcf    joy         0.75
2 ed1gxcf    joy         0.75
3 ed1gxcf    neutral     0.20
4 ed1gxcf    neutral     0.20
```

This confirms the module correctly:

- Reads the parquet file
- Selects child emotion columns
- Maps emotions → macro category
- Computes valence

### Step 2 - Run full emotion aggregation pipeline

```bash
python run_emotion_aggregation.py
```

**Expected terminal output:**

```bash
Loaded parent_child_pairs.parquet with shape: (2530, 71)
Child results shape: (2530, 3)
Saved child emotion scores to: data/emotion_scores_child.parquet
Parent results shape: (2530, 3)
Saved parent emotion scores to: data/emotion_scores_parent.parquet
```

### Step 3 - Verify output files

```bash
python check_emotion_scores.py
```

**Expected terminal output:**

```bash
Child scores shape: (2530, 3)
Parent scores shape: (2530, 3)
```

# Part 3: Emotional Contagion Analysis  
**Author:** Kaylee Cameron  
**Component:** Emotion Propagation Modeling (Pipeline Stage 3)

---

## 3.1 Overview

This module analyzes how emotions propagate within Reddit reply threads by combining:

- Parent–child emotion labels  
- Continuous valence scores  
- Thread depth information  

The goal is to quantify whether emotional tone systematically transfers from parent comments to replies, and how this influence changes across thread depth.

---

## 3.2 Core Analytic Outputs

This stage produces the following metrics and artifacts:

### 3.2.1 Parent → Child Valence Correlation  
- Pearson and Spearman correlations for valence alignment  
- Summary saved to:  
  `emocon/results/valence_correlation.json`

### 3.2.2 Emotion Transition Probabilities  
- Full transition matrix across emotion categories  
- Probability that child emotion matches or shifts relative to parent  
- Saved to:  
  `emocon/results/emotion_transitions.json`

### 3.2.3 Propagation Strength Ranking  
- Per-emotion contagion strength: *Which emotions carry forward most?*  
- Saved to:  
  `emocon/results/propagation_strength.json`

### 3.2.4 Depth-Based Decay Modeling  
- Analysis of how emotional matching changes from depth 1 → depth 2  
- Decay plots and fitted model parameters  
- Saved to:  
  `emocon/figures/decay_curve.png`  
  `emocon/results/decay_stats.json`

### 3.2.5 Statistical Significance Testing  
- Chi-square test for independence between parent & child emotion  
- Z-test comparing emotional matching across depths  
- Saved to:  
  `emocon/results/significance_stats.json`

### 3.2.6 Outlier Thread Identification  
- Detection of threads where emotion propagation is unusually strong  
- Saved to:  
  `emocon/results/outlier_analysis.json`

---

## 3.3 Code Structure

emocon/
└── contagion/
├── model.py
├── emotion_transitions.py
├── propagation_strength.py
├── decay_model.py
├── significance_tests.py
└── outlier_analysis.py
