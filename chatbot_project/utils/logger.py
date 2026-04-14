# =========================================================================
# File Name: utils/logger.py
# Version: 5.2.0 (Smart Logging — Rich Context, VRAM Tracking, Timing)
# Project: Absher Smart Assistant (MOI ChatBot)
# Features:
# - Dual Output: Color console + detailed file (with function:line)
# - JSON Structured Log: Machine-readable for post-mortem analysis
# - System Banner: Logs GPU, Python, VRAM, OS info on first init
# - VRAM Snapshots: log_vram() helper for tracking GPU memory
# - Timing Decorator: @timed for automatic latency measurement
# - Session Context: Thread-local user/request tracking
# - Rotation: Nightly rotation, 30-day retention, UTF-8 Arabic safe
# =========================================================================
import logging
import sys
import os
import time
import json
import platform
import threading
import functools
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import Optional, Callable
from config import Config

# =========================================================================
# 1. SESSION CONTEXT (Thread-Local User Tracking)
# =========================================================================
_session_context = threading.local()

def set_session_context(username: str = "guest", request_id: str = ""):
    """Set user context for current thread — appears in all subsequent logs."""
    _session_context.username = username
    _session_context.request_id = request_id

def get_session_context() -> dict:
    return {
        "username": getattr(_session_context, 'username', 'system'),
        "request_id": getattr(_session_context, 'request_id', ''),
    }

# =========================================================================
# 2. COLORED CONSOLE FORMATTER
# =========================================================================
class ColoredFormatter(logging.Formatter):
    """
    Emoji + ANSI color console output. Pre-compiled for performance.
    Shows: [Level] [Module] Message
    """
    GREY = "\x1b[38;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    CYAN = "\x1b[36;20m"
    RESET = "\x1b[0m"

    def __init__(self):
        super().__init__()
        self._formatters = {
            logging.DEBUG: logging.Formatter(f"{self.GREY}\U0001f41b DEBUG: %(message)s{self.RESET}"),
            logging.INFO: logging.Formatter(f"{self.GREEN}\U0001f4a1 INFO: %(message)s{self.RESET}"),
            logging.WARNING: logging.Formatter(f"{self.YELLOW}\u26a0\ufe0f WARNING: %(message)s{self.RESET}"),
            logging.ERROR: logging.Formatter(f"{self.RED}\u274c ERROR: %(message)s{self.RESET}"),
            logging.CRITICAL: logging.Formatter(f"{self.BOLD_RED}\U0001f6a8 CRITICAL: %(message)s{self.RESET}"),
        }
        self._default = logging.Formatter("%(message)s")

    def format(self, record: logging.LogRecord) -> str:
        formatter = self._formatters.get(record.levelno, self._default)
        return formatter.format(record)


# =========================================================================
# 3. STRUCTURED JSON FORMATTER (for machine-readable log file)
# =========================================================================
class JSONFormatter(logging.Formatter):
    """
    Outputs one JSON object per line — ideal for log aggregation tools,
    grep/jq analysis, and post-mortem debugging.
    """
    def format(self, record: logging.LogRecord) -> str:
        ctx = get_session_context()
        entry = {
            "ts": datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            "level": record.levelname,
            "module": record.name,
            "func": record.funcName,
            "line": record.lineno,
            "msg": record.getMessage(),
            "user": ctx["username"],
        }
        if ctx["request_id"]:
            entry["req_id"] = ctx["request_id"]
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


# =========================================================================
# 4. DETAILED FILE FORMATTER (human-readable with full trace)
# =========================================================================
class DetailedFormatter(logging.Formatter):
    """
    Rich file format: Timestamp | Module | Function:Line | Level | User | Message
    """
    def format(self, record: logging.LogRecord) -> str:
        ctx = get_session_context()
        user_tag = f"[{ctx['username']}]" if ctx['username'] != 'system' else ""
        base = (
            f"{datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')} "
            f"- [{record.name}] - [{record.funcName}:{record.lineno}] "
            f"- {record.levelname} {user_tag} - {record.getMessage()}"
        )
        if record.exc_info and record.exc_info[0]:
            base += "\n" + self.formatException(record.exc_info)
        return base


# =========================================================================
# 5. SYSTEM BANNER (logged once on first initialization)
# =========================================================================
_banner_logged = False

def _log_system_banner(logger: logging.Logger):
    """Logs hardware, software, and configuration info on startup."""
    global _banner_logged
    if _banner_logged:
        return
    _banner_logged = True

    lines = [
        "=" * 60,
        "\U0001f1f8\U0001f1e6  ABSHER SMART ASSISTANT — SYSTEM INFO",
        "=" * 60,
        f"  Python:     {platform.python_version()}",
        f"  Platform:   {platform.system()} {platform.release()} ({platform.machine()})",
        f"  Node:       {platform.node()}",
        f"  PID:        {os.getpid()}",
    ]

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram_used = torch.cuda.memory_allocated() / 1e9
            lines.append(f"  GPU:        {gpu_name}")
            lines.append(f"  VRAM:       {vram_used:.1f} / {vram_total:.1f} GB")
            lines.append(f"  CUDA:       {torch.version.cuda}")
            lines.append(f"  PyTorch:    {torch.__version__}")
            lines.append(f"  Precision:  {Config.TORCH_DTYPE}")
        else:
            lines.append("  GPU:        None (CPU mode)")
    except ImportError:
        lines.append("  GPU:        torch not available")

    lines.append(f"  LLM:        {getattr(Config, 'LLM_MODEL_NAME', 'N/A')}")
    lines.append(f"  Embeddings: {getattr(Config, 'EMBEDDING_MODEL_NAME', 'N/A')}")
    lines.append(f"  T-S-T:      {'Enabled (NLLB-200)' if getattr(Config, 'TST_ENABLED', False) else 'Disabled'}")
    lines.append(f"  Debug:      {getattr(Config, 'DEBUG_MODE', False)}")
    lines.append("=" * 60)

    for line in lines:
        logger.info(line)


# =========================================================================
# 6. VRAM SNAPSHOT HELPER
# =========================================================================
def log_vram(logger: logging.Logger, label: str = ""):
    """
    Logs current GPU memory state. Call anywhere for debugging.
    Usage: log_vram(logger, "after loading ALLaM")
    """
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            free = total - allocated
            tag = f" [{label}]" if label else ""
            logger.info(
                f"\U0001f4ca VRAM{tag}: "
                f"Allocated={allocated:.1f}GB | Reserved={reserved:.1f}GB | "
                f"Free={free:.1f}GB | Total={total:.1f}GB"
            )
    except Exception:
        pass


# =========================================================================
# 7. TIMING DECORATOR
# =========================================================================
def timed(logger_name: str = "Absher_AI"):
    """
    Decorator that logs function execution time.
    Usage:
        @timed("RAG_Engine")
        def run(self, query):
            ...
    Logs: "⏱️ run() completed in 3.42s"
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logging.getLogger(logger_name)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                _logger.info(f"\u23f1\ufe0f {func.__name__}() completed in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                _logger.error(f"\u23f1\ufe0f {func.__name__}() FAILED after {elapsed:.2f}s: {e}")
                raise
        return wrapper
    return decorator


# =========================================================================
# 8. MAIN LOGGER FACTORY
# =========================================================================
def setup_logger(name: str = "Absher_AI", log_filename: str = "app.log") -> logging.Logger:
    """
    Creates a logger with 3 handlers:
      1. Console: Color-coded, emoji, message-only (sysadmin UX)
      2. File (detailed): Timestamp, module, function:line, user context
      3. File (JSON): Machine-readable structured logs for analysis

    Args:
        name: Logger identifier (e.g., "RAG_Engine", "Translator")
        log_filename: Name of the human-readable log file

    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(name)

    # Singleton guard: don't add duplicate handlers
    if logger.handlers:
        return logger

    log_level = logging.DEBUG if getattr(Config, 'DEBUG_MODE', False) else logging.INFO
    logger.setLevel(log_level)

    # ── Handler 1: Console (colored, concise) ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    # ── Handler 2: Detailed file (human-readable) ──
    try:
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        log_path = os.path.join(Config.LOG_DIR, log_filename)

        file_handler = TimedRotatingFileHandler(
            log_path, when="midnight", interval=1,
            backupCount=30, encoding='utf-8'
        )
        file_handler.setFormatter(DetailedFormatter())
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"\u26a0\ufe0f File logging disabled: {e}")

    # ── Handler 3: JSON structured log ──
    try:
        json_log_path = os.path.join(Config.LOG_DIR, "structured.jsonl")
        json_handler = TimedRotatingFileHandler(
            json_log_path, when="midnight", interval=1,
            backupCount=14, encoding='utf-8'
        )
        json_handler.setFormatter(JSONFormatter())
        json_handler.setLevel(logging.INFO)  # JSON only captures INFO+
        logger.addHandler(json_handler)
    except Exception as e:
        print(f"\u26a0\ufe0f JSON logging disabled: {e}")

    # Prevent log bubbling to root
    logger.propagate = False

    # Log system banner on first logger creation
    _log_system_banner(logger)

    return logger