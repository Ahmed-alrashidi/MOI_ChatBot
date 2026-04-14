# =========================================================================
# File Name: utils/telemetry.py
# Purpose: User Analytics & Audit Trail Logging.
# Version: 5.1.1 (Added matched_services for persistent memory)
# Features:
# - Captures user queries, AI responses, latency, and hardware/browser info.
# - [NEW v5.1] Logs user feedback (👍/👎) for quality monitoring.
# - [NEW v5.1.1] Stores matched_services so rag_pipeline can restore
#   last_service on server restart (Fix #1 from analysis report).
# =========================================================================

import os
import re
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

from config import Config

TELEMETRY_DIR = os.path.join(Config.OUTPUTS_DIR, "user_analytics")


def log_interaction(
    username: str,
    query: str,
    response: str,
    latency: float,
    client_ip: str,
    user_agent: str,
    matched_services: Optional[List[str]] = None
):
    """
    Saves a detailed record of a single chat interaction for a specific user.
    Appends the record to a JSON Lines (.jsonl) file named after the user.

    Args:
        username: The logged-in user ID.
        query: The user's question.
        response: The AI-generated response.
        latency: Time taken in seconds.
        client_ip: The user's IP address.
        user_agent: Browser/OS info.
        matched_services: [NEW v5.1.1] List of KG service names matched
            during this query. Stored so memory can restore last_service
            after server restart.
    """
    try:
        os.makedirs(TELEMETRY_DIR, exist_ok=True)

        safe_username = username if username else "guest_user"
        safe_username = re.sub(r'[^a-zA-Z0-9_\-\u0600-\u06FF]', '_', safe_username)
        user_file = os.path.join(TELEMETRY_DIR, f"{safe_username}_history.jsonl")

        record: Dict[str, Any] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "unix_time": time.time(),
            "username": safe_username,
            "client_ip": client_ip,
            "browser_os": user_agent,
            "latency_seconds": round(latency, 2),
            "user_query": query,
            "ai_response": response,
            "matched_services": matched_services or []
        }

        with open(user_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    except Exception as e:
        print(f"⚠️ Telemetry Error: Failed to log interaction for {username}: {e}")


def log_feedback(
    username: str,
    query: str,
    response: str,
    rating: str
):
    """
    [NEW v5.1] Logs user feedback (positive/negative) for quality monitoring.
    Stored in a separate _feedback.jsonl file per user.
    """
    try:
        os.makedirs(TELEMETRY_DIR, exist_ok=True)

        safe_username = username if username else "guest_user"
        safe_username = re.sub(r'[^a-zA-Z0-9_\-\u0600-\u06FF]', '_', safe_username)
        feedback_file = os.path.join(TELEMETRY_DIR, f"{safe_username}_feedback.jsonl")

        record: Dict[str, Any] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "unix_time": time.time(),
            "username": safe_username,
            "feedback": rating,
            "user_query": query,
            "ai_response": response[:300]
        }

        with open(feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    except Exception as e:
        print(f"⚠️ Feedback Error: Failed to log feedback for {username}: {e}")