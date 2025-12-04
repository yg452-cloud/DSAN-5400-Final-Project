# data/download_data.py
"""
Download GoEmotions Dataset
===========================

Downloads the GoEmotions dataset from Hugging Face.

Usage:
    python download_data.py

Output:
    - goemotions_local.csv (~80-100 MB)
    
Logs:
    - ../logs/data_acquisition.log
"""

from loader import RedditDataLoader
from logging_config import setup_logging
import logging
import sys

def main():
    # Setup logging
    logger = setup_logging(logging.INFO)
    
    logger.info("=" * 70)
    logger.info("GoEmotions Dataset Download")
    logger.info("=" * 70)
    logger.info("Download parameters:")
    logger.info("  Source: Hugging Face")
    logger.info("  URL: https://huggingface.co/datasets/mrm8488/goemotions")
    logger.info("  Target: goemotions_local.csv")
    logger.info("")
    
    print()
    print("=" * 70)
    print("        GoEmotions Dataset Downloader")
    print("=" * 70)
    print()
    print("This will download ~80-100 MB from Hugging Face")
    print()
    
    try:
        filepath = RedditDataLoader.download_from_huggingface()
        
        logger.info("=" * 70)
        logger.info(f"Download successful: {filepath}")
        logger.info("=" * 70)
        logger.info("")
        
        print()
        print("=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print(f"\nData saved to: {filepath}")
        print(f"Logs saved to: ../logs/data_acquisition.log")
        print()
        print("Next step:")
        print("  python run_full_pipeline.py")
        print()
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Download cancelled by user")
        print("\nCancelled")
        return 1
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
        print("\nManual download:")
        print("  1. Visit: https://huggingface.co/datasets/mrm8488/goemotions")
        print("  2. Download goemotions.csv")
        print("  3. Save as: goemotions_local.csv")
        return 1

if __name__ == "__main__":
    sys.exit(main())