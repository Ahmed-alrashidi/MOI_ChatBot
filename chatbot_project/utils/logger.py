import logging
import sys
import os
# We need to append the project root to sys.path to import config if run as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

def setup_logger(name=__name__):
    """
    Configures and returns a logger instance.
    It writes logs to a file (app.log) and prints them to the console.
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate logs if handler already exists
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # 1. File Handler (Saves logs to disk)
        try:
            file_handler = logging.FileHandler(Config.LOG_FILE, mode='a', encoding='utf-8')
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"⚠️ Warning: Could not create log file handler: {e}")
        
        # 2. Stream Handler (Prints to terminal)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_formatter = logging.Formatter('%(levelname)s: %(message)s')
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)
        
    return logger