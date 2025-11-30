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