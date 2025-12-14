"""
Command-line interface for the Emocon package.

This CLI provides access to all major pipeline stages:
1. Data acquisition and preprocessing
2. Emotion aggregation
3. Contagion analysis
4. Full pipeline execution
"""

import click
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from emocon.data.pipeline import run_data_pipeline
from emocon.data.loader import RedditDataLoader
from emocon.models.emotion_model import EmotionAggregator
from emocon.contagion.model import load_and_merge_data
from emocon.utils import setup_logging
import pandas as pd

# Member 3 contagion analysis modules
from emocon.contagion import analysis as contagion_analysis
from emocon.contagion import propogation_strength
from emocon.contagion import decay_model
from emocon.contagion import emotion_transitions
from emocon.contagion import significance_tests
from emocon.contagion import outlier_analysis
from emocon.visualization.plotter import plot_average_emotion_probs, plot_depth_valence_correlation, plot_emotion_barplot, plot_emotion_corr_heatmap, plot_parent_child_valence_scatter, plot_valence_hist
# Member 5 visualization module
try:
    from emocon.visualization import plotter
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


@click.group()
@click.version_option(version="0.1.0")
def main():
    """
    Emocon: Emotional Contagion Analysis for Reddit Comment Threads

    A tool for analyzing how emotions propagate through online discussions.
    """
    pass


@main.command()
@click.option(
    "--output-dir",
    default="data",
    help="Directory to save downloaded data",
    type=click.Path(),
)
def download(output_dir):
    """Download the GoEmotions dataset from Hugging Face."""
    click.echo("=" * 70)
    click.echo("Downloading GoEmotions Dataset")
    click.echo("=" * 70)

    output_path = Path(output_dir) / "goemotions_local.csv"

    try:
        RedditDataLoader.download_from_huggingface(str(output_path))
        click.echo(f"\n Dataset downloaded to: {output_path}")
    except Exception as e:
        click.echo(f"\n Download failed: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--min-depth",
    default=1,
    help="Minimum thread depth to retain",
    type=int,
)
def preprocess(min_depth):
    """Run data preprocessing and thread graph construction."""
    click.echo("=" * 70)
    click.echo("Data Preprocessing & Thread Graph Construction")
    click.echo("=" * 70)

    try:
        logger = setup_logging()
        run_data_pipeline()
        click.echo("\n Preprocessing complete!")
        click.echo("  Output files:")
        click.echo("    - data/threads_with_replies.parquet")
        click.echo("    - data/parent_child_pairs.parquet")
    except Exception as e:
        click.echo(f"\n Preprocessing failed: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--input-file",
    default="data/parent_child_pairs.parquet",
    help="Input parquet file with parent-child pairs",
    type=click.Path(exists=True),
)
@click.option(
    "--output-dir",
    default="data",
    help="Directory to save emotion scores",
    type=click.Path(),
)
def aggregate_emotions(input_file, output_dir):
    """Aggregate fine-grained emotions into macro categories and valence scores."""
    click.echo("=" * 70)
    click.echo("Emotion Aggregation")
    click.echo("=" * 70)

    try:
        # Load data
        click.echo(f"Loading data from: {input_file}")
        df = pd.read_parquet(input_file)
        click.echo(f"  Loaded {len(df):,} parent-child pairs")

        # Process child emotions
        click.echo("\nProcessing child emotions...")
        child_aggregator = EmotionAggregator(role="child")
        child_results = child_aggregator.process_dataframe(df)

        child_output = Path(output_dir) / "emotion_scores_child.parquet"
        child_results.to_parquet(child_output, index=False)
        click.echo(f"   Saved: {child_output}")

        # Process parent emotions
        click.echo("\nProcessing parent emotions...")
        parent_aggregator = EmotionAggregator(role="parent")
        parent_results = parent_aggregator.process_dataframe(df)

        parent_output = Path(output_dir) / "emotion_scores_parent.parquet"
        parent_results.to_parquet(parent_output, index=False)
        click.echo(f"   Saved: {parent_output}")

        click.echo("\n Emotion aggregation complete!")

    except Exception as e:
        click.echo(f"\n Emotion aggregation failed: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--output-file",
    default="data/contagion_ready.parquet",
    help="Output file for merged contagion dataset",
    type=click.Path(),
)
def prepare_contagion(output_file):
    """Merge emotion scores with thread data for contagion analysis."""
    click.echo("=" * 70)
    click.echo("Preparing Contagion Dataset")
    click.echo("=" * 70)

    try:
        click.echo("Merging parent-child pairs with emotion scores...")
        df = load_and_merge_data()

        click.echo(f"  Dataset shape: {df.shape}")
        click.echo(f"  Columns: {', '.join(df.columns)}")

        # Save
        df.to_parquet(output_file, index=False)
        click.echo(f"\n Saved contagion dataset to: {output_file}")

    except Exception as e:
        click.echo(f"\n Contagion preparation failed: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--skip-download",
    is_flag=True,
    help="Skip data download step",
)
@click.option(
    "--skip-preprocess",
    is_flag=True,
    help="Skip preprocessing step",
)
def analyze(skip_download, skip_preprocess):
    """Run the complete analysis pipeline (all stages)."""
    click.echo("=" * 70)
    click.echo("EMOCON: Full Pipeline Execution")
    click.echo("=" * 70)

    # Stage 1: Download
    if not skip_download:
        click.echo("\n[Stage 1/5] Downloading dataset...")
        data_file = Path("data/goemotions_local.csv")
        if data_file.exists():
            click.echo(f"   Dataset already exists: {data_file}")
        else:
            try:
                RedditDataLoader.download_from_huggingface()
            except Exception as e:
                click.echo(f"   Download failed: {str(e)}", err=True)
                sys.exit(1)

    # Stage 2: Preprocess
    if not skip_preprocess:
        click.echo("\n[Stage 2/5] Preprocessing data...")
        try:
            logger = setup_logging()
            run_data_pipeline()
            click.echo("   Preprocessing complete")
        except Exception as e:
            click.echo(f"   Preprocessing failed: {str(e)}", err=True)
            sys.exit(1)

    # Stage 3: Aggregate emotions
    click.echo("\n[Stage 3/5] Aggregating emotions...")
    try:
        df = pd.read_parquet("data/parent_child_pairs.parquet")

        child_aggregator = EmotionAggregator(role="child")
        child_results = child_aggregator.process_dataframe(df)
        child_results.to_parquet("data/emotion_scores_child.parquet", index=False)

        parent_aggregator = EmotionAggregator(role="parent")
        parent_results = parent_aggregator.process_dataframe(df)
        parent_results.to_parquet("data/emotion_scores_parent.parquet", index=False)

        click.echo("   Emotion aggregation complete")
    except Exception as e:
        click.echo(f"   Emotion aggregation failed: {str(e)}", err=True)
        sys.exit(1)

    # Stage 4: Prepare contagion dataset
    click.echo("\n[Stage 4/5] Preparing contagion analysis dataset...")
    try:
        contagion_df = load_and_merge_data()
        contagion_df.to_parquet("data/contagion_ready.parquet", index=False)
        click.echo("   Contagion dataset ready")
    except Exception as e:
        click.echo(f"   Contagion preparation failed: {str(e)}", err=True)
        sys.exit(1)

    # Stage 5: Run contagion analysis (Member 3)
    click.echo("\n[Stage 5/5] Running emotional contagion analysis...")
    try:
        import os
        import json
        os.makedirs("results", exist_ok=True)
        os.makedirs("figures", exist_ok=True)

        # Load contagion data
        df = pd.read_parquet("data/contagion_ready.parquet")
        click.echo(f"   Loaded contagion data: {df.shape}")

        # 1. Basic contagion statistics
        click.echo("   Computing valence contagion...")
        contagion_stats = contagion_analysis.compute_valence_contagion(df)
        with open("results/contagion_stats.json", "w") as f:
            json.dump(contagion_stats, f, indent=4)

        # 2. Propagation strength analysis
        click.echo("   Computing propagation strength...")
        prop_results = propogation_strength.compute_propagation_strength(df)
        propogation_strength.save_results(prop_results)
        propogation_strength.plot_propagation(prop_results)

        # 3. Decay model
        click.echo("   Computing decay model...")
        decay_df = decay_model.compute_depth_decay(df)
        slope_stats = decay_model.simple_slope(decay_df)
        decay_model.plot_decay(decay_df)
        decay_model.save_stats(slope_stats, decay_df)

        # 4. Emotion transitions
        click.echo("   Analyzing emotion transitions...")
        transition_counts, transition_probs = emotion_transitions.build_transition_matrix(df)
        emotion_transitions.save_results(transition_counts, transition_probs)
        emotion_transitions.plot_heatmap(transition_probs)

        # 5. Significance tests
        click.echo("   Running significance tests...")
        chi2_result = significance_tests.chi_square_test(
            significance_tests.compute_transition_matrix(df)
        )
        depth_results = significance_tests.compute_depth_significance(df)
        sig_results = {"chi_square": chi2_result, "depth_analysis": depth_results}
        significance_tests.save_results(sig_results)

        # 6. Outlier analysis
        click.echo("   Detecting outliers...")
        parent_grouped = outlier_analysis.compute_parent_propagation(df)
        outliers = outlier_analysis.identify_outliers(parent_grouped)
        outlier_analysis.save_results(outliers)

        click.echo("   Contagion analysis complete!")

        # 7. Additional visualizations (Member 5)
        if VISUALIZATION_AVAILABLE:
            click.echo("   Generating additional visualizations...")
            try:
                # Member 5: Add your visualization functions here
                # Example:
                # plotter.plot_emotion_distribution(df)
                # plotter.plot_thread_depth_analysis(df)
                # plotter.create_summary_dashboard(df)
                pass
            except Exception as viz_error:
                click.echo(f"   Warning: Visualization failed: {viz_error}")

    except Exception as e:
        click.echo(f"   Contagion analysis failed: {str(e)}", err=True)
        click.echo("   (This is non-fatal, continuing...)")


    # Summary
    click.echo("\n" + "=" * 70)
    click.echo("PIPELINE COMPLETE!")
    click.echo("=" * 70)
    click.echo("\nGenerated data files:")
    click.echo("  1. data/threads_with_replies.parquet")
    click.echo("  2. data/parent_child_pairs.parquet")
    click.echo("  3. data/emotion_scores_child.parquet")
    click.echo("  4. data/emotion_scores_parent.parquet")
    click.echo("  5. data/contagion_ready.parquet")
    click.echo("\nGenerated analysis results:")
    click.echo("  - results/*.json (contagion stats, decay, transitions, significance tests)")
    click.echo("  - figures/*.png (visualizations)")
    click.echo("\nLogs: logs/data_acquisition.log")
    click.echo("=" * 70)


@main.command()
def info():
    """Display package and data information."""
    click.echo("=" * 70)
    click.echo("Emocon Package Information")
    click.echo("=" * 70)
    click.echo(f"Project: Emotional Contagion Analysis")
    click.echo(f"Course: DSAN-5400 Natural Language Processing")
    click.echo("\nAuthors:")
    click.echo("  - Ke Tian")
    click.echo("  - Kaylee Cameron")
    click.echo("  - Matthew Hakim")
    click.echo("  - Yanmin Gui")
    click.echo("  - Jiaheng Cao")

    click.echo("\n" + "=" * 70)
    click.echo("Data Files Status")
    click.echo("=" * 70)

    files_to_check = [
        "data/goemotions_local.csv",
        "data/threads_with_replies.parquet",
        "data/parent_child_pairs.parquet",
        "data/emotion_scores_child.parquet",
        "data/emotion_scores_parent.parquet",
        "data/contagion_ready.parquet",
    ]

    for file_path in files_to_check:
        p = Path(file_path)
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            click.echo(f"   {file_path} ({size_mb:.2f} MB)")
        else:
            click.echo(f"   {file_path} (not found)")

    click.echo("\n" + "=" * 70)


if __name__ == "__main__":
    main()
