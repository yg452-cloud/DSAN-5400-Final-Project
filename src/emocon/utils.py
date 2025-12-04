# data/logging_config.py
"""
Logging configuration for the data acquisition pipeline.

Ensures all logs are saved to ../logs/data_acquisition.log for reproducibility.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logging(log_level=logging.INFO):
    """
    Setup logging configuration for data acquisition.
    
    Creates:
    - Console handler (prints to terminal)
    - File handler (saves to ../logs/data_acquisition.log)
    
    Args:
        log_level: Logging level (default: INFO)
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "data_acquisition.log"
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter('%(message)s')
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler (simple format)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (detailed format)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Log session start
    logger.info("=" * 70)
    logger.info(f"Logging session: {datetime.now()}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 70)
    
    return logger


if __name__ == "__main__":
    """Test logging configuration"""
    logger = setup_logging()
    
    logger.info("Testing INFO message")
    logger.warning("Testing WARNING message")
    logger.error("Testing ERROR message")
    
    print("\nLogging test complete!")
    print("Check: ../logs/data_acquisition.log")