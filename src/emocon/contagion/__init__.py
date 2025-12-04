"""
Emotional contagion analysis module.

Components:
- model: Data merging and contagion dataset construction
- emotion_transitions: Emotion transition matrices
- propogation_strength: Per-emotion contagion ranking
- decay_model: Depth-based decay analysis
- significance_tests: Statistical tests
- outlier_analysis: Extreme-case emotional pattern detection
- analysis: Main analysis orchestration
"""

from .model import load_and_merge_data

__all__ = ["load_and_merge_data"]
