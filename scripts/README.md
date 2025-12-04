# Development Scripts

This directory contains utility scripts for package development and maintenance.

## Available Scripts

### `verify_package.py`
Comprehensive verification script that checks if the package is properly configured.

**Usage:**
```bash
python scripts/verify_package.py
```

**What it checks:**
- ✓ Configuration files (pyproject.toml, environment.yml)
- ✓ Package structure (all __init__.py files)
- ✓ Test suite existence
- ✓ Documentation files
- ✓ Package imports
- ✓ CLI commands
- ✓ Data files (optional)

**Example output:**
```
======================================================================
EMOCON PACKAGE VERIFICATION
======================================================================
...
Checks passed: 18/18
✓ ALL CHECKS PASSED! Package is properly configured.
```

---

## Adding New Scripts

When adding development scripts, follow these guidelines:

1. **Name clearly**: Use descriptive names (e.g., `clean_cache.py`, `update_deps.py`)
2. **Add docstrings**: Document what the script does
3. **Add to this README**: Describe usage and purpose
4. **Make executable** (optional): `chmod +x script_name.py`

### Example script template:

```python
#!/usr/bin/env python
"""
Brief description of what this script does.
"""

import sys
from pathlib import Path

def main():
    """Main function."""
    # Your code here
    pass

if __name__ == "__main__":
    sys.exit(main())
```

---

## Common Development Tasks

### Run tests
```bash
pytest tests/ -v
```

### Format code
```bash
black src/emocon/
```

### Check code style
```bash
flake8 src/emocon/
```

### Build package
```bash
pip install build
python -m build
```

### Install in development mode
```bash
pip install -e .
```
