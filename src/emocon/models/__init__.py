"""
Emotion modeling and classification module.

Components:
- emotion_model: Emotion aggregation from fine-grained GoEmotions labels
"""

from .emotion_model import EmotionAggregator

__all__ = ["EmotionAggregator"]
