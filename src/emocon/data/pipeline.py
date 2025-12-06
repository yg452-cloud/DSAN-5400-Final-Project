# data/run_full_pipeline.py
"""
Part 1: Data Acquisition & Thread Graph Construction Pipeline
================================================================

This script runs the complete data processing pipeline:
1. Load GoEmotions dataset
2. Remove rater duplicates (aggregate by comment ID)
3. Clean text
4. Build thread graphs
5. Extract parent-child pairs
6. Save results

Usage:
    python run_full_pipeline.py

Outputs:
    - parent_child_pairs.parquet
    - threads_with_replies.parquet
    
Logs:
    - All operations logged to ../logs/data_acquisition.log
"""

from loader import RedditDataLoader
from text_cleaner import TextCleaner
from thread_builder import ThreadBuilder
from logging_config import setup_logging
import logging
import pandas as pd

# Setup logging (saves to both console and file)
logger = setup_logging(logging.INFO)

logger.info("=" * 70)
logger.info("Part 1: Data Acquisition & Thread Graph Construction")
logger.info("=" * 70)

# Log pipeline parameters
logger.info("Pipeline Parameters:")
logger.info("  Data source: goemotions_local.csv (GoEmotions dataset)")
logger.info("  Rater deduplication: Enabled (aggregate before processing)")
logger.info("  Min thread depth: 1")
logger.info("  Text cleaning: Enabled")
logger.info("  Output format: Parquet")
logger.info("")

print("=== Part 1 Complete Pipeline ===\n")

# Step 1: Load data
print("Step 1: Loading data...")
logger.info("Step 1: Loading data")
loader = RedditDataLoader("goemotions_local.csv")
df_raw = loader.load()
stats = loader.get_basic_stats()

logger.info(f"Raw data loaded: {len(df_raw):,} rows (includes multiple raters)")
logger.info(f"Unique comments: {df_raw['id'].nunique():,}")

# Step 2: Remove rater duplicates
print("\nStep 2: Removing rater duplicates...")
logger.info("Step 2: Aggregating multiple rater annotations")

# Define emotion columns
EMOTION_COLS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

def aggregate_raters(df):
    """
    Aggregate multiple rater annotations for each comment.
    
    Args:
        df: DataFrame with multiple rows per comment (one per rater)
        
    Returns:
        DataFrame with one row per comment (averaged emotions)
    """
    def agg_comment(group):
        result = {}
        
        # Keep metadata (should be same for all raters)
        result['id'] = group['id'].iloc[0]
        result['text'] = group['text'].iloc[0]
        result['parent_id'] = group['parent_id'].iloc[0]
        result['link_id'] = group['link_id'].iloc[0]
        
        # Keep other metadata if present
        for col in ['author', 'subreddit', 'created_utc']:
            if col in group.columns:
                result[col] = group[col].iloc[0]
        
        # Average emotion labels across raters
        for emotion in EMOTION_COLS:
            if emotion in group.columns:
                result[emotion] = group[emotion].mean()
        
        # Track number of raters
        result['num_raters'] = len(group)
        
        return pd.Series(result)
    
    df_agg = df.groupby('id', as_index=False).apply(agg_comment)
    df_agg = df_agg.reset_index(drop=True)
    
    return df_agg

logger.info(f"Before aggregation: {len(df_raw):,} rows")
df = aggregate_raters(df_raw)
logger.info(f"After aggregation: {len(df):,} unique comments")
logger.info(f"Removed: {len(df_raw) - len(df):,} duplicate rows")
logger.info(f"Average raters per comment: {df['num_raters'].mean():.2f}")

# Verify deduplication
assert df['id'].nunique() == len(df), "ERROR: Duplicate comment IDs still present"
logger.info("Verification: All comment IDs are unique")

# Step 3: Clean text
print("\nStep 3: Cleaning text...")
logger.info("Step 3: Text cleaning")
df = TextCleaner.clean_dataframe(df)

# Step 4: Build threads
print("\nStep 4: Building thread graphs...")
logger.info("Step 4: Building thread graphs")
builder = ThreadBuilder(df)
builder.build_thread_graphs()
df = builder.calculate_depths()

# Step 5: Filter threads
print("\nStep 5: Filtering threads with replies...")
logger.info("Step 5: Filtering threads (min_depth=1)")
df_with_replies = builder.filter_deep_threads(min_depth=1)

# Step 6: Extract pairs
print("\nStep 6: Extracting parent-child pairs...")
logger.info("Step 6: Extracting parent-child pairs")
pairs = builder.get_parent_child_pairs()

# Verify no duplicate pairs
unique_pairs = pairs[['id_parent', 'id_child']].drop_duplicates()
if len(pairs) != len(unique_pairs):
    logger.warning(f"WARNING: Found {len(pairs) - len(unique_pairs)} duplicate pairs")
else:
    logger.info(f"Verification: All {len(pairs):,} pairs are unique")

# Step 7: Save results
print("\nStep 7: Saving results...")
logger.info("Step 7: Saving results")

df_with_replies.to_parquet('threads_with_replies.parquet', index=False)
logger.info(f"  Saved: threads_with_replies.parquet ({len(df_with_replies):,} rows)")

pairs.to_parquet('parent_child_pairs.parquet', index=False)
logger.info(f"  Saved: parent_child_pairs.parquet ({len(pairs):,} rows)")

# Log final statistics
logger.info("")
logger.info("=" * 70)
logger.info("Pipeline Completed Successfully")
logger.info("=" * 70)
logger.info("")
logger.info("Processing Summary:")
logger.info(f"  Raw data (with rater duplicates): {len(df_raw):,} rows")
logger.info(f"  After deduplication: {len(df):,} unique comments")
logger.info(f"  After text cleaning: {len(df):,} comments")
logger.info(f"  In filtered threads: {len(df_with_replies):,} comments")
logger.info(f"  Parent-child pairs: {len(pairs):,}")
logger.info("")

logger.info("Data Quality:")
logger.info(f"  All comment IDs unique: {df['id'].nunique() == len(df)}")
logger.info(f"  All pairs unique: {len(pairs) == len(unique_pairs)}")
logger.info("")

logger.info("Thread Statistics:")
logger.info(f"  Threads with replies: {df_with_replies['link_id'].nunique():,}")
logger.info(f"  Unique parents: {pairs['id_parent'].nunique():,}")
logger.info(f"  Unique children: {pairs['id_child'].nunique():,}")
logger.info("")

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
print(f"Raw data: {len(df_raw):,} rows (with rater duplicates)")
print(f"After deduplication: {len(df):,} unique comments")
print(f"Threads with replies: {df_with_replies['link_id'].nunique():,}")
print(f"Parent-child pairs: {len(pairs):,}")
print(f"\nData Quality:")
print(f"  All comment IDs unique: {df['id'].nunique() == len(df)}")
print(f"  All pairs unique: {len(pairs) == len(unique_pairs)}")
print(f"\nFiles saved:")
print(f"  - threads_with_replies.parquet")
print(f"  - parent_child_pairs.parquet")
print(f"\nLogs saved to: ../logs/data_acquisition.log")
print()