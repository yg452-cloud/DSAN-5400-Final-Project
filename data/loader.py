# data/loader.py

import pandas as pd
import logging
from pathlib import Path
from typing import Optional
import urllib.request
import os

logger = logging.getLogger(__name__)

class RedditDataLoader:
    """
    Load Reddit GoEmotions dataset from Hugging Face or local file.
    
    Workflow:
    1. First time: Download data using download_from_huggingface()
    2. Then: Load from local file using load()
    """
    
    # Hugging Face dataset URL
    HF_URL = "https://huggingface.co/datasets/mrm8488/goemotions/resolve/main/goemotions.csv"
    DEFAULT_FILENAME = "goemotions_local.csv"
    
    def __init__(self, source: str = DEFAULT_FILENAME):
        """
        Initialize the data loader.
        
        Args:
            source: Local CSV file path (default: goemotions_local.csv)
        """
        self.source = source
        self.data: Optional[pd.DataFrame] = None
        logger.info(f"Initialized RedditDataLoader with source: {source}")
    
    @classmethod
    def download_from_huggingface(cls, save_path: str = None) -> str:
        """
        Download GoEmotions dataset from Hugging Face to current directory.
        
        Args:
            save_path: Where to save (default: goemotions_local.csv in current dir)
            
        Returns:
            Path to the downloaded file
            
        Example:
            # First time: Download
            RedditDataLoader.download_from_huggingface()
            
            # Then: Load
            loader = RedditDataLoader()
            df = loader.load()
        """
        if save_path is None:
            save_path = cls.DEFAULT_FILENAME
        
        logger.info("=" * 60)
        logger.info("Downloading GoEmotions dataset from Hugging Face")
        logger.info("=" * 60)
        logger.info(f"Source: {cls.HF_URL}")
        logger.info(f"Destination: {save_path}")
        
        try:
            # Check if file already exists
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path) / (1024 * 1024)
                logger.warning(f"File already exists: {save_path} ({file_size:.2f} MB)")
                logger.info("Skipping download. Delete the file if you want to re-download.")
                return save_path
            
            # Download with progress indicator
            logger.info("Download started... (this may take a few minutes)")
            
            def _show_progress(block_num, block_size, total_size):
                """Show download progress"""
                downloaded = block_num * block_size
                percent = min(int(downloaded * 100 / total_size), 100)
                if block_num % 100 == 0:  # Update every 100 blocks
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    logger.info(f"Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            
            urllib.request.urlretrieve(cls.HF_URL, save_path, _show_progress)
            
            # Verify download
            file_size = os.path.getsize(save_path) / (1024 * 1024)
            logger.info("=" * 60)
            logger.info("Download complete!")
            logger.info(f"File saved to: {save_path}")
            logger.info(f"File size: {file_size:.2f} MB")
            logger.info("=" * 60)
            
            return save_path
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error("Download failed!")
            logger.error(f"Error: {str(e)}")
            logger.error("=" * 60)
            logger.error("Manual download instructions:")
            logger.error("1. Go to: https://huggingface.co/datasets/mrm8488/goemotions")
            logger.error("2. Download goemotions.csv")
            logger.error(f"3. Save as: {save_path}")
            raise
    
    def load(self) -> pd.DataFrame:
        """
        Load the dataset from local CSV file.
        
        Returns:
            DataFrame containing Reddit comments with emotion labels
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        try:
            # Check if file exists
            if not os.path.exists(self.source):
                logger.error(f"File not found: {self.source}")
                logger.error("")
                logger.error("Please download the dataset first:")
                logger.error("from loader import RedditDataLoader")
                logger.error("RedditDataLoader.download_from_huggingface()")
                logger.error("")
                raise FileNotFoundError(
                    f"File not found: {self.source}\n"
                    f"Run RedditDataLoader.download_from_huggingface() first."
                )
            
            logger.info(f"Loading data from {self.source}...")
            self.data = pd.read_csv(self.source)
            
            logger.info(f"Successfully loaded {len(self.data):,} comments")
            logger.info(f"Columns ({len(self.data.columns)}): {', '.join(self.data.columns[:5])}...")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def get_basic_stats(self) -> dict:
        """
        Get basic statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        stats = {
            "total_comments": len(self.data),
            "comments_with_parent": self.data['parent_id'].notna().sum(),
            "unique_threads": self.data['link_id'].nunique() if 'link_id' in self.data.columns else 0,
            "num_columns": len(self.data.columns),
            "columns": list(self.data.columns)
        }
        
        logger.info("Dataset Statistics:")
        logger.info(f"  Total comments: {stats['total_comments']:,}")
        logger.info(f"  Comments with parent: {stats['comments_with_parent']:,}")
        logger.info(f"  Unique threads: {stats['unique_threads']:,}")
        
        return stats


if __name__ == "__main__":
    """
    Demo usage of the loader.
    """
    import logging
    
    # Setup nice logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    print("\n" + "=" * 60)
    print("RedditDataLoader Demo")
    print("=" * 60 + "\n")
    
    # Check if data file exists
    if not os.path.exists(RedditDataLoader.DEFAULT_FILENAME):
        print("Data file not found. Downloading...")
        print()
        RedditDataLoader.download_from_huggingface()
        print()
    
    # Load the data
    print("Loading data from local file...")
    print()
    loader = RedditDataLoader()
    df = loader.load()
    
    # Show sample
    print("\n" + "=" * 60)
    print("Sample Data")
    print("=" * 60)
    print(df[['text', 'id', 'parent_id']].head(3))
    
    # Show statistics
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    stats = loader.get_basic_stats()
    
    print("\nLoader demo complete!\n")