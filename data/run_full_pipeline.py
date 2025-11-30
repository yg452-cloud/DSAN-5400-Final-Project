# data/run_full_pipeline.py
"""
Part 1: Data Acquisition & Thread Graph Construction Pipeline
================================================================

This script runs the complete data processing pipeline:
1. Load GoEmotions dataset
2. Clean text
3. Build thread graphs
4. Extract parent-child pairs
5. Save results

Usage:
    python run_full_pipeline.py

Outputs:
    - parent_child_pairs.parquet (2530 pairs)
    - threads_with_replies.parquet
    
Logs:
    - All operations logged to ../logs/data_acquisition.log
"""

from loader import RedditDataLoader
from text_cleaner import TextCleaner
from thread_builder import ThreadBuilder
from logging_config import setup_logging
import logging

# Setup logging (saves to both console and file)
logger = setup_logging(logging.INFO)

logger.info("=" * 70)
logger.info("Part 1: Data Acquisition & Thread Graph Construction")
logger.info("=" * 70)

# Log pipeline parameters
logger.info("Pipeline Parameters:")
logger.info("  Data source: goemotions_local.csv (GoEmotions dataset)")
logger.info("  Min thread depth: 1 (adjusted from original requirement)")
logger.info("  Text cleaning: Enabled")
logger.info("  Output format: Parquet")
logger.info("")

print("=== Part 1 Complete Pipeline ===\n")

# Step 1: Load data
print("Step 1: Loading data...")
logger.info("Step 1: Loading data")
loader = RedditDataLoader("goemotions_local.csv")
df = loader.load()
stats = loader.get_basic_stats()

# Step 2: Clean text
print("\nStep 2: Cleaning text...")
logger.info("Step 2: Text cleaning")
df = TextCleaner.clean_dataframe(df)

# Step 3: Build threads
print("\nStep 3: Building thread graphs...")
logger.info("Step 3: Building thread graphs")
builder = ThreadBuilder(df)
builder.build_thread_graphs()
df = builder.calculate_depths()

# Step 4: Filter threads (using depth >= 1 instead of 3)
print("\nStep 4: Filtering threads with replies...")
logger.info("Step 4: Filtering threads (min_depth=1)")
df_with_replies = builder.filter_deep_threads(min_depth=1)

# Step 5: Extract pairs
print("\nStep 5: Extracting parent-child pairs...")
logger.info("Step 5: Extracting parent-child pairs")
pairs = builder.get_parent_child_pairs()

# Step 6: Save results
print("\nStep 6: Saving results...")
logger.info("Step 6: Saving results")

df_with_replies.to_parquet('threads_with_replies.parquet', index=False)
logger.info(f"  Saved: threads_with_replies.parquet ({len(df_with_replies):,} rows)")

pairs.to_parquet('parent_child_pairs.parquet', index=False)
logger.info(f"  Saved: parent_child_pairs.parquet ({len(pairs):,} rows)")

# Log final statistics
logger.info("")
logger.info("=" * 70)
logger.info("Pipeline Completed Successfully")
logger.info("=" * 70)
logger.info("Final Statistics:")
logger.info(f"  Total comments processed: {len(df):,}")
logger.info(f"  Threads with replies: {df_with_replies['link_id'].nunique():,}")
logger.info(f"  Comments in threads: {len(df_with_replies):,}")
logger.info(f"  Parent-child pairs: {len(pairs):,}")
logger.info("")

# Log depth distribution
logger.info("Depth Distribution:")
depth_dist = df['depth'].value_counts().sort_index()
for depth, count in depth_dist.items():
    logger.info(f"  Depth {depth}: {count:,} comments")
logger.info("")

logger.info("Output Files:")
logger.info("  - threads_with_replies.parquet")
logger.info("  - parent_child_pairs.parquet")
logger.info("=" * 70)
logger.info("")

# Console output
print("\n=== Summary ===")
print(f"Total comments: {len(df):,}")
print(f"Threads with replies: {df_with_replies['link_id'].nunique():,}")
print(f"Comments in threads: {len(df_with_replies):,}")
print(f"Parent-child pairs: {len(pairs):,}")
print(f"\nFiles saved:")
print(f"- threads_with_replies.parquet")
print(f"- parent_child_pairs.parquet")
print(f"\nLogs saved to: ../logs/data_acquisition.log")
print()