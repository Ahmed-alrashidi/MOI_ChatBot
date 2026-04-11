# =========================================================================
# File Name: utils/logger.py
# Purpose: Advanced Logging System with Traceability (Line numbers & Functions).
# Project: Absher Smart Assistant (MOI ChatBot)
# Features:
# - Level Management: Toggleable via Config.DEBUG_MODE.
# - Detailed Auditing: Captures function names and line numbers in file logs.
# - Visual Console: Clean, color-coded UX for real-time monitoring.
# - Optimized: Pre-compiled formatters for high-throughput performance.
# =========================================================================

import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler
from config import Config

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter for the console output. 
    Provides emoji-based, color-coded feedback while keeping the output concise. 
    Formatters are pre-compiled in __init__ to save CPU cycles during high-frequency logging.
    """
    # ANSI Color escape sequences for terminal output
    GREY = "\x1b[38;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"
    
    # Message-only format for a clean, user-friendly terminal experience
    BASE_FMT = "%(message)s"

    def __init__(self):
        super().__init__()
        # Pre-compile formatters to avoid redundant string concatenation per log event
        self._formatters = {
            logging.DEBUG: logging.Formatter(f"{self.GREY}🐛 DEBUG: {self.BASE_FMT}{self.RESET}"),
            logging.INFO: logging.Formatter(f"{self.GREEN}💡 INFO: {self.BASE_FMT}{self.RESET}"),
            logging.WARNING: logging.Formatter(f"{self.YELLOW}⚠️ WARNING: {self.BASE_FMT}{self.RESET}"),
            logging.ERROR: logging.Formatter(f"{self.RED}❌ ERROR: {self.BASE_FMT}{self.RESET}"),
            logging.CRITICAL: logging.Formatter(f"{self.BOLD_RED}🚨 CRITICAL: {self.BASE_FMT}{self.RESET}")
        }
        self._default_formatter = logging.Formatter(self.BASE_FMT)

    def format(self, record: logging.LogRecord) -> str:
        """Applies the mapped color formatter to the intercepted log record."""
        formatter = self._formatters.get(record.levelno, self._default_formatter)
        return formatter.format(record)


def setup_logger(name: str = "Absher_AI", log_filename: str = "app.log") -> logging.Logger:
    """
    Orchestrates the logging infrastructure with dual-target dispatching:
    1. File Handler: Comprehensive audit trail (Timestamp, Module, Line, Level) for debugging.
    2. Console Handler: Streamlined, color-coded status updates for the sysadmin.
    
    Args:
        name (str): Identifier for the logger instance.
        log_filename (str): Name of the output log file.
        
    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Singleton Guard: Prevent redundant handler attachment if called multiple times
    if logger.handlers:
        return logger

    # 1. Determine Global Sensitivity Level based on Environment Config
    log_level = logging.DEBUG if getattr(Config, 'DEBUG_MODE', False) else logging.INFO
    logger.setLevel(log_level)

    # 2. File Handler Configuration (The Detailed Auditor)
    try:
        if not os.path.exists(Config.LOG_DIR):
            os.makedirs(Config.LOG_DIR, exist_ok=True)

        log_path = os.path.join(Config.LOG_DIR, log_filename)

        # DETAILED FORMAT: [Timestamp] - [Module] - [Function:Line] - [Level] - [Message]
        # Crucial for tracing asynchronous bugs on the A100 cluster.
        detailed_formatter = logging.Formatter(
            '%(asctime)s - [%(name)s] - [%(funcName)s:%(lineno)d] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Rotates logs nightly, keeping a 30-day history. UTF-8 ensures Arabic logs don't break.
        file_handler = TimedRotatingFileHandler(
            log_path, 
            when="midnight", 
            interval=1, 
            backupCount=30, 
            encoding='utf-8' 
        )
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
    except (OSError, PermissionError) as e:
        print(f"⚠️ Warning: Persistent file logging disabled (Permission Denied): {e}")
    except Exception as e:
        print(f"⚠️ Warning: Unexpected file logging initialization failure: {e}")

    # 3. Console Handler Configuration (The Visual Monitor)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)

    # Prevent logs from bubbling up to the root logger (prevents duplicate console prints)
    logger.propagate = False
    
    return logger