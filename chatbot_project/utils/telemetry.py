# =========================================================================
# File Name: utils/telemetry.py
# Purpose: User Analytics & Audit Trail Logging.
# Features:
# - Captures user queries, AI responses, latency, and hardware/browser info.
# - Stores data incrementally in JSONL format per user for easy analytics.
# - Automatically creates user-specific directories in the outputs folder.
# =========================================================================

import os
import re
import json
import time
from datetime import datetime
from typing import Dict, Any

# Ensure paths are loaded correctly
from config import Config

# Define the central directory for user analytics within the outputs folder
TELEMETRY_DIR = os.path.join(Config.OUTPUTS_DIR, "user_analytics")

def log_interaction(
    username: str, 
    query: str, 
    response: str, 
    latency: float, 
    client_ip: str, 
    user_agent: str
):
    """
    Saves a detailed record of a single chat interaction for a specific user.
    Appends the record to a JSON Lines (.jsonl) file named after the user.
    
    Args:
        username (str): The ID of the logged-in user.
        query (str): The text or transcribed audio requested by the user.
        response (str): The final AI-generated response.
        latency (float): Time taken to generate the response in seconds.
        client_ip (str): The IP address of the user's device.
        user_agent (str): Browser and OS information.
    """
    try:
        # 1. Ensure the telemetry directory exists
        os.makedirs(TELEMETRY_DIR, exist_ok=True)
        
        # 2. Define the user's specific log file
        # Fallback to 'guest' if the username is somehow missing
        safe_username = username if username else "guest_user"
        # Sanitize username for filesystem safety (prevent path traversal)
        safe_username = re.sub(r'[^a-zA-Z0-9_\-\u0600-\u06FF]', '_', safe_username)
        user_file = os.path.join(TELEMETRY_DIR, f"{safe_username}_history.jsonl")
        
        # 3. Construct the telemetry payload
        record: Dict[str, Any] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "unix_time": time.time(),
            "username": safe_username,
            "client_ip": client_ip,
            "browser_os": user_agent,
            "latency_seconds": round(latency, 2),
            "user_query": query,
            "ai_response": response
        }
        
        # 4. Append the record to the file safely (UTF-8 for Arabic support)
        with open(user_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    except Exception as e:
        # Fail silently to prevent crashing the main chat interface
        print(f"⚠️ Telemetry Error: Failed to log interaction for {username}: {e}")