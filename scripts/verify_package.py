#!/usr/bin/env python
"""
Quick verification script for the Emocon package.
Run this to verify that all packaging components are in place.
"""

import sys
from pathlib import Path
import subprocess

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def check_file(filepath, description):
    """Check if a file exists."""
    p = Path(filepath)
    if p.exists():
        size = p.stat().st_size
        print(f"✓ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"✗ {description}: {filepath} (NOT FOUND)")
        return False


def check_import(module_name, description):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {description}: {module_name} - {e}")
        return False


def run_command(cmd, description):
    """Run a command and check if it succeeds."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"✓ {description}")
            return True
        else:
            print(f"✗ {description} - Exit code: {result.returncode}")
            return False
    except Exception as e:
        print(f"✗ {description} - {e}")
        return False


def main():
    print("=" * 70)
    print("EMOCON PACKAGE VERIFICATION")
    print("=" * 70)
    print()

    all_checks = []

    # Configuration files
    print("1. Configuration Files")
    print("-" * 70)
    all_checks.append(
        check_file("pyproject.toml", "Package configuration")
    )
    all_checks.append(
        check_file("environment.yml", "Conda environment")
    )
    print()

    # Package structure
    print("2. Package Structure")
    print("-" * 70)
    all_checks.append(
        check_file("src/emocon/__init__.py", "Main package init")
    )
    all_checks.append(
        check_file("src/emocon/cli.py", "CLI module")
    )
    all_checks.append(
        check_file("src/emocon/utils.py", "Utils module")
    )
    all_checks.append(
        check_file("src/emocon/data/__init__.py", "Data module init")
    )
    all_checks.append(
        check_file("src/emocon/models/__init__.py", "Models module init")
    )
    all_checks.append(
        check_file("src/emocon/contagion/__init__.py", "Contagion module init")
    )
    all_checks.append(
        check_file("src/emocon/visualization/__init__.py", "Visualization module init")
    )
    print()

    # Tests
    print("3. Tests")
    print("-" * 70)
    all_checks.append(
        check_file("tests/test_pipeline.py", "Test suite")
    )
    print()

    # Documentation
    print("4. Documentation")
    print("-" * 70)
    all_checks.append(
        check_file("README.md", "README")
    )
    all_checks.append(
        check_file("PACKAGING_SUMMARY.md", "Packaging summary")
    )
    print()

    # Imports
    print("5. Package Imports")
    print("-" * 70)
    all_checks.append(
        check_import("emocon", "Main package")
    )
    all_checks.append(
        check_import("emocon.data.loader", "Data loader")
    )
    all_checks.append(
        check_import("emocon.models.emotion_model", "Emotion model")
    )
    all_checks.append(
        check_import("emocon.contagion.model", "Contagion model")
    )
    print()

    # CLI commands
    print("6. CLI Commands")
    print("-" * 70)
    all_checks.append(
        run_command("emocon --version", "CLI version command")
    )
    all_checks.append(
        run_command("emocon info", "CLI info command")
    )
    print()

    # Data files
    print("7. Data Files (Optional - may not exist yet)")
    print("-" * 70)
    check_file("data/goemotions_local.csv", "GoEmotions dataset")
    check_file("data/parent_child_pairs.parquet", "Parent-child pairs")
    check_file("data/emotion_scores_child.parquet", "Child emotion scores")
    check_file("data/emotion_scores_parent.parquet", "Parent emotion scores")
    check_file("data/contagion_ready.parquet", "Contagion dataset")
    print()

    # Summary
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    passed = sum(all_checks)
    total = len(all_checks)
    print(f"Checks passed: {passed}/{total}")

    if passed == total:
        print("\n✓ ALL CHECKS PASSED! Package is properly configured.")
        return 0
    else:
        print(f"\n✗ {total - passed} check(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
