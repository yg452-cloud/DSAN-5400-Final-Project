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

from .loader import RedditDataLoader
from .text_cleaner import TextCleaner
from .thread_builder import ThreadBuilder
from ..utils import setup_logging 
import logging
import os
from pathlib import Path

def run_data_pipeline():
    # Setup logging (saves to both console and file)
    logger = setup_logging()
    logger.info("Starting Data Pipeline...")

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

    logger.info("=== Part 1 Complete Pipeline ===\n")

    # Step 1: Load data
    logger.info("Step 1: Loading data...")

    _CURRENT_DIR = Path(__file__).parent.resolve() 
    _PROJECT_ROOT = _CURRENT_DIR.parent.parent.parent  
    _DATA_DIR = _PROJECT_ROOT / "data"
    loader = RedditDataLoader(str(_DATA_DIR / "goemotions_local.csv"))
    df = loader.load()
    stats = loader.get_basic_stats()

    # Step 2: Clean text
    logger.info("\nStep 2: Cleaning text...")
    df = TextCleaner.clean_dataframe(df)

    # Step 3: Build threads
    logger.info("\nStep 3: Building thread graphs...")
    builder = ThreadBuilder(df)
    builder.build_thread_graphs()
    df = builder.calculate_depths()

    # Step 4: Filter threads (using depth >= 1 instead of 3)
    logger.info("\nStep 4: Filtering threads with replies...")
    df_with_replies = builder.filter_deep_threads(min_depth=1)

    # Step 5: Extract pairs
    logger.info("\nStep 5: Extracting parent-child pairs...")
    pairs = builder.get_parent_child_pairs()

    # Step 6: Save results
    logger.info("\nStep 6: Saving results...")


    df_with_replies.to_parquet(str(_DATA_DIR/'threads_with_replies.parquet'), index=False)
    logger.info(f"  Saved: threads_with_replies.parquet ({len(df_with_replies):,} rows)")

    pairs.to_parquet(str(_DATA_DIR/'parent_child_pairs.parquet'), index=False)
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
    logger.info("\n=== Summary ===")
    logger.info(f"Total comments: {len(df):,}")
    logger.info(f"Threads with replies: {df_with_replies['link_id'].nunique():,}")
    logger.info(f"Comments in threads: {len(df_with_replies):,}")
    logger.info(f"Parent-child pairs: {len(pairs):,}")
    logger.info(f"\nFiles saved:")
    logger.info(f"- threads_with_replies.parquet")
    logger.info(f"- parent_child_pairs.parquet")
    logger.info(f"\nLogs saved to: logs/data_acquisition.log")
    logger.info("")


if __name__ == "__main__":
    """Test pipeline """
    print("--- DEBUG MODE: Testing Pipeline Paths ---")
    run_data_pipeline()