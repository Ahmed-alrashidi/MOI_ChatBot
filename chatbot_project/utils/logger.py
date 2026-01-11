import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler

# Temporarily append parent directory to path to import Config safely
# This allows running scripts from subdirectories (e.g., python data/ingestion.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import Config

def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Configures and returns a production-ready logger instance with Log Rotation.
    
    Features:
    - **Log Rotation:** Creates a new log file every midnight (backup count: 30 days).
    - **Dual Output:** Detailed logs to file, clean logs to console.
    - **UTF-8 Support:** Critical for Arabic logs.
    """
    logger = logging.getLogger(name)
    
    # Prevent adding multiple handlers if setup is called multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # --- Formatters ---
        # File: Very Detailed (Time + Module + Func + Line + Level + Message)
        file_formatter = logging.Formatter(
            '%(asctime)s - [%(name)s:%(funcName)s:%(lineno)d] - %(levelname)s - %(message)s'
        )
        # Console: Clean (Level + Message only)
        stream_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # --- 1. Rotating File Handler (Production Safe) ---
        try:
            log_dir = os.path.dirname(Config.LOG_FILE)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            # Rotate logs every midnight, keep last 30 days
            file_handler = TimedRotatingFileHandler(
                Config.LOG_FILE, 
                when="midnight", 
                interval=1, 
                backupCount=30, 
                encoding='utf-8'
            )
            file_handler.setFormatter(file_formatter)
            file_handler.suffix = "%Y-%m-%d" # Suffix for archived logs
            logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"⚠️ Warning: Could not create log file handler: {e}")
        
        # --- 2. Stream Handler (Console) ---
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)
        
        # Stop propagation to root logger to avoid duplicates
        logger.propagate = False
        
    return logger