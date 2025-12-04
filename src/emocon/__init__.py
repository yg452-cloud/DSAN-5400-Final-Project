"""
Emocon: Emotional Contagion Analysis for Reddit Comment Threads
================================================================

A Python package for analyzing emotional contagion in online discussions.
Part of the DSAN-5400 Final Project: "Echo to Empathy"

Main Components:
- data: Data acquisition, preprocessing, and thread graph construction
- models: Emotion classification and aggregation
- contagion: Emotional propagation analysis and modeling
- visualization: Data visualization and insight generation
"""

__version__ = "0.1.0"
__author__ = "Ke Tian, Kaylee Cameron, Matthew Hakim, Yanmin Gui, Jiaheng Cao"

# Import key classes for easy access
from .models.emotion_model import EmotionAggregator
from .data.loader import RedditDataLoader
from .utils import setup_logging

__all__ = [
    "EmotionAggregator",
    "RedditDataLoader",
    "setup_logging",
]
