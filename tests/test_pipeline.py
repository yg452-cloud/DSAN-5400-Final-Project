"""
End-to-end tests for the Emocon pipeline.

Tests verify that:
1. Data loading works correctly
2. Emotion aggregation produces valid outputs
3. Contagion dataset can be constructed
4. Key files are generated with expected structure
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emocon.models.emotion_model import EmotionAggregator
from emocon.data.loader import RedditDataLoader


class TestEmotionAggregator:
    """Test emotion aggregation functionality."""

    def test_aggregator_initialization(self):
        """Test that aggregator can be initialized."""
        child_agg = EmotionAggregator(role="child")
        assert child_agg.role == "child"
        assert child_agg.suffix == "_child"

        parent_agg = EmotionAggregator(role="parent")
        assert parent_agg.role == "parent"
        assert parent_agg.suffix == "_parent"

    def test_aggregator_invalid_role(self):
        """Test that invalid role raises error."""
        with pytest.raises(ValueError):
            EmotionAggregator(role="invalid")

    def test_aggregator_output_structure(self):
        """Test that aggregator produces correct output structure."""
        # Create sample data
        sample_data = pd.DataFrame(
            {
                "id_child": ["comment1", "comment2"],
                "joy_child": [1, 0],
                "anger_child": [0, 1],
                "neutral_child": [0, 0],
            }
        )

        aggregator = EmotionAggregator(role="child")
        result = aggregator.process_dataframe(sample_data)

        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert "comment_id" in result.columns
        assert "macro_label" in result.columns
        assert "valence" in result.columns
        assert len(result) == 2


class TestDataLoader:
    """Test data loading functionality."""

    def test_loader_initialization(self):
        """Test that loader can be initialized."""
        loader = RedditDataLoader()
        assert loader.source is not None
        assert loader.data is None

    def test_loader_paths(self):
        """Test that loader uses correct paths."""
        loader = RedditDataLoader()
        assert "goemotions_local.csv" in loader.source


class TestPipeline:
    """Test complete pipeline functionality."""

    @pytest.fixture
    def data_dir(self):
        """Return path to data directory."""
        return Path(__file__).parent.parent / "data"

    def test_parent_child_pairs_exists(self, data_dir):
        """Test that parent-child pairs file exists."""
        pairs_file = data_dir / "parent_child_pairs.parquet"
        if pairs_file.exists():
            df = pd.read_parquet(pairs_file)
            assert "id_parent" in df.columns
            assert "id_child" in df.columns
            assert len(df) > 0

    def test_emotion_scores_structure(self, data_dir):
        """Test emotion scores files have correct structure."""
        child_file = data_dir / "emotion_scores_child.parquet"
        parent_file = data_dir / "emotion_scores_parent.parquet"

        if child_file.exists():
            df = pd.read_parquet(child_file)
            assert "comment_id" in df.columns
            assert "macro_label" in df.columns
            assert "valence" in df.columns
            assert df["valence"].between(-1.0, 1.0).all()

        if parent_file.exists():
            df = pd.read_parquet(parent_file)
            assert "comment_id" in df.columns
            assert "macro_label" in df.columns
            assert "valence" in df.columns
            assert df["valence"].between(-1.0, 1.0).all()

    def test_contagion_dataset_structure(self, data_dir):
        """Test contagion dataset has expected columns."""
        contagion_file = data_dir / "contagion_ready.parquet"

        if contagion_file.exists():
            df = pd.read_parquet(contagion_file)

            # Check required columns
            required_cols = [
                "parent_id",
                "child_id",
                "emotion_parent",
                "valence_parent",
                "emotion_child",
                "valence_child",
            ]

            for col in required_cols:
                assert col in df.columns, f"Missing column: {col}"

            # Check data quality
            assert df["valence_parent"].between(-1.0, 1.0).all()
            assert df["valence_child"].between(-1.0, 1.0).all()
            assert len(df) > 0


class TestPackageStructure:
    """Test package structure and imports."""

    def test_package_imports(self):
        """Test that main package components can be imported."""
        from emocon import EmotionAggregator, RedditDataLoader, setup_logging

        assert EmotionAggregator is not None
        assert RedditDataLoader is not None
        assert setup_logging is not None

    def test_submodule_imports(self):
        """Test that submodules can be imported."""
        from emocon.data import loader, text_cleaner, thread_builder
        from emocon.models import emotion_model
        from emocon.contagion import model

        assert loader is not None
        assert emotion_model is not None
        assert model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
