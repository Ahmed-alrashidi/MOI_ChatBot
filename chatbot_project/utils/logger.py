import logging
import sys
import os
from typing import Optional

# Temporarily append parent directory to path to import Config safely
# This is necessary because 'utils' is a subdirectory, and Python needs to see the root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Configures and returns a production-ready logger instance.
    
    Design Philosophy:
    - **Dual Output:** Writes detailed logs to a file (for debugging) and concise logs 
      to the console (for monitoring).
    - **Idempotency:** Checks if handlers exist to prevent duplicate log entries.
    - **Resilience:** Includes a fallback mechanism if file permissions deny writing logs.

    Args:
        name (str): The name of the logger (typically __name__ of the calling module).

    Returns:
        logging.Logger: A configured logger object ready for use.
    """
    logger = logging.getLogger(name)
    
    # --- Check for existing handlers ---
    # This prevents the "Duplicate Log" issue where the same line prints multiple times
    # if setup_logger is called repeatedly.
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # --- Formatters ---
        # File: Detailed (Time + Module + Level + Message)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        # Console: Clean (Level + Message only)
        stream_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # --- 1. File Handler (Persistent Storage) ---
        try:
            # Ensure the directory for logs exists; otherwise, FileHandler will crash.
            log_dir = os.path.dirname(Config.LOG_FILE)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            # Create handler to append mode ('a') with UTF-8 support for Arabic text
            file_handler = logging.FileHandler(Config.LOG_FILE, mode='a', encoding='utf-8')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
        except Exception as e:
            # Critical Fallback: If the filesystem is Read-Only or permission is denied,
            # we warn the user but do NOT crash the app.
            print(f"⚠️ Warning: Could not create log file handler at {Config.LOG_FILE}: {e}")
        
        # --- 2. Stream Handler (Console Output) ---
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)
        
        # --- Propagation Control ---
        # Stop logs from bubbling up to the root logger.
        # This ensures our formatting rules are strictly applied and not overridden by global settings.
        logger.propagate = False
        
    return logger