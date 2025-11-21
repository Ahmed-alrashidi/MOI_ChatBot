#!/bin/bash

echo "🚀 Starting MOI Chatbot Setup..."

# 1. Install System Dependencies (ffmpeg is required for Whisper)
# Check if we are on Linux/Colab (apt-get exists)
if command -v apt-get &> /dev/null; then
    echo "🔹 Installing system dependencies (ffmpeg)..."
    # Use sudo if not root, unless on Colab where user is root
    if [ "$EUID" -ne 0 ]; then
        sudo apt-get update -qq && sudo apt-get install -y ffmpeg -qq
    else
        apt-get update -qq && apt-get install -y ffmpeg -qq
    fi
else
    echo "⚠️ Warning: 'apt-get' not found. If on Windows/Mac, please install ffmpeg manually."
fi

# 2. Upgrade pip
echo "🔹 Upgrading pip..."
pip install --upgrade pip -q

# 3. Install Python Libraries
echo "🔹 Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

# 4. Create a template .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "🔹 Creating template .env file..."
    echo "HF_TOKEN=paste_your_huggingface_token_here" > .env
    echo "⚠️ A '.env' file has been created. Please open it and paste your HF Token inside!"
else
    echo "✅ .env file already exists."
fi

echo "✅ Setup Complete! You can now run: python main.py"