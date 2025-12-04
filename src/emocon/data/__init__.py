"""
Data acquisition and preprocessing module.

Components:
- loader: Download and load Reddit GoEmotions dataset
- text_cleaner: Text preprocessing and cleaning utilities
- thread_builder: Thread graph construction and parent-child pair extraction
- pipeline: Complete data processing pipeline
"""

from .loader import RedditDataLoader
from .text_cleaner import TextCleaner
from .thread_builder import ThreadBuilder
from .pipeline import run_data_pipeline

__all__ = [
    "RedditDataLoader",
    "TextCleaner",
    "ThreadBuilder",
    "run_data_pipeline",
]
