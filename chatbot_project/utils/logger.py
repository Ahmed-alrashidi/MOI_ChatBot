import logging
import sys
import os
from typing import Optional

# Append project root to sys.path to allow importing config safely
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Configures and returns a standardized logger instance.
    
    Features:
    - Dual output: File (app.log) and Console (stdout).
    - Prevents duplicate handlers if called multiple times.
    - Handles permission errors gracefully.
    
    Args:
        name (str): Name of the logger (usually __name__).
        
    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger(name)
    
    # Check if handlers already exist to prevent duplicate logs
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        stream_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # 1. File Handler (Saves logs to disk)
        try:
            # Ensure log directory exists explicitly before creating handler
            log_dir = os.path.dirname(Config.LOG_FILE)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(Config.LOG_FILE, mode='a', encoding='utf-8')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
        except Exception as e:
            # Fallback if file cannot be written (e.g., read-only file system)
            print(f"⚠️ Warning: Could not create log file handler at {Config.LOG_FILE}: {e}")
        
        # 2. Stream Handler (Prints to terminal)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)
        
        # Prevent propagation to root logger to avoid double printing in some envs
        logger.propagate = False
        
    return logger