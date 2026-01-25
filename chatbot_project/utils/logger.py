# =========================================================================
# File Name: utils/logger.py
# Purpose: Advanced Logging System with Dynamic Levels & Error Handling.
# Project: Absher Smart Assistant (MOI ChatBot)
# Features:
# - Level Control: Synchronized with Config.DEBUG_MODE.
# - Fault Tolerance: Gracefully handles directory permission errors.
# - Visual UX: Color-coded console outputs for easier debugging.
# - Log Maintenance: Automatic daily rotation with 30-day retention.
# =========================================================================

import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler
from config import Config

class ColoredFormatter(logging.Formatter):
    """
    Custom logging formatter that injects ANSI escape sequences to provide 
    color-coded feedback in the console based on the importance of the message.
    """
    # ANSI Color Codes for terminal output
    GREY = "\x1b[38;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"
    
    FORMAT = "%(message)s"

    # Mapping log levels to specific colors and symbols (e.g., Emoji for Debug)
    FORMATS = {
        logging.DEBUG: GREY + "🐛 DEBUG: " + FORMAT + RESET,
        logging.INFO: GREEN + "INFO: " + FORMAT + RESET,
        logging.WARNING: YELLOW + "WARNING: " + FORMAT + RESET,
        logging.ERROR: RED + "ERROR: " + FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + "CRITICAL: " + FORMAT + RESET
    }

    def format(self, record):
        """Formats the log record with the appropriate level-based color."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(name: str = __name__, log_filename: str = "app.log") -> logging.Logger:
    """
    Orchestrates the creation and configuration of the logger instance.
    Ensures that multiple handlers (File and Console) are attached correctly 
    without duplication.

    Args:
        name (str): The name of the module requesting the logger.
        log_filename (str): The physical filename for the log output.

    Returns:
        logging.Logger: A fully configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Singleton check: Prevent adding duplicate handlers if the logger already exists
    if logger.handlers:
        return logger

    # --- 1. Dynamic Logging Level Selection ---
    # Switches to DEBUG (verbose) if Config.DEBUG_MODE is enabled, else uses INFO.
    level = logging.DEBUG if getattr(Config, 'DEBUG_MODE', False) else logging.INFO
    logger.setLevel(level)

    # --- 2. File Handler Configuration (Permanent Storage) ---
    try:
        # Self-Healing: Ensure the log directory exists before attempting to write
        if not os.path.exists(Config.LOG_DIR):
            os.makedirs(Config.LOG_DIR, exist_ok=True)

        log_path = os.path.join(Config.LOG_DIR, log_filename)

        # Standard non-colored format for file logs (includes timestamps and module names)
        file_formatter = logging.Formatter(
            '%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File Rotation: Creates a new file at midnight and keeps logs for 30 days.
        file_handler = TimedRotatingFileHandler(
            log_path, 
            when="midnight", 
            interval=1, 
            backupCount=30, 
            encoding='utf-8' # Ensures Arabic text is written correctly
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
    except (OSError, PermissionError) as e:
        # Security Guard: If the server denies write access, fallback to console-only logging
        print(f"⚠️ Warning: Logging to file disabled due to permissions: {e}")
    except Exception as e:
        print(f"⚠️ Warning: Unexpected error initializing file logger: {e}")

    # --- 3. Console Handler Configuration (Real-time Monitoring) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)

    # Prevent logs from bubbling up to the root logger to avoid double-logging
    logger.propagate = False
    return logger