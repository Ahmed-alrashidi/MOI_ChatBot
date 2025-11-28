"""
ui/theme.py

This module contains the visual definitions for the Gradio User Interface.
It strictly separates the 'Presentation Layer' (CSS/HTML) from the 'Logic Layer' (Python).

Key Components:
1. Custom CSS for MOI branding (Green/Gold theme, Tajawal font).
2. HTML structure for the custom header.
3. Asset paths for logos.
"""

import os

# Path to the logo image. 
# Note: 'file/' prefix is specific to Gradio to serve local files securely.
LOGO_PATH: str = "file/ui/moi_logo.png"

# --- CSS Styling ---
# Defines the look and feel. 
# - Imports 'Tajawal' font for modern Arabic typography.
# - Overrides default Gradio styles to match Ministry of Interior colors.
MOI_CSS: str = """
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');

/* Global Font & Background Settings */
body, .gradio-container { 
    font-family: 'Tajawal', sans-serif !important; 
    background-color: #f0f2f5 !important; 
}

/* --- Tech Header Container --- */
.moi-header {
    display: flex;
    justify-content: space-between; /* Spacing between Title and Tech Badges */
    align-items: center;
    padding: 15px 40px; 
    /* MOI Official Gradient (Dark Green) */
    background: linear-gradient(135deg, #004D26 0%, #006C35 100%);
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(0, 77, 38, 0.2);
    margin-bottom: 20px;
    border-bottom: 4px solid #d4af37; /* Gold accent line */
}

/* --- Logo & Title Group (Right Side / RTL) --- */
.header-title-group {
    display: flex;
    flex-direction: row-reverse; /* Ensure Logo is to the right of text in RTL */
    align-items: center;
    gap: 20px;
    text-align: right;
}

.moi-logo {
    width: 80px;
    height: auto;
    filter: drop-shadow(0 4px 6px rgba(0,0,0,0.3));
}

.header-title-group h1 { 
    color: #ffffff !important; 
    font-size: 1.8em;
    margin: 0; 
    font-weight: 700;
}

.header-title-group p { 
    color: #d4af37 !important; /* Gold text */
    font-size: 0.9em;
    margin: 0; 
    opacity: 0.9;
}

/* --- Tech Status Badges (Left Side) --- */
.tech-badges {
    display: flex;
    gap: 10px;
    align-items: center; 
}

.badge {
    background: rgba(255, 255, 255, 0.15); /* Glassmorphism effect */
    padding: 4px 12px;
    border-radius: 20px;
    color: #fff;
    font-size: 0.8em;
    font-family: monospace;
    border: 1px solid rgba(255, 255, 255, 0.2);
    display: flex;
    align-items: center;
    gap: 5px;
}

/* Glowing Green Dot for 'Online' status */
.dot {
    height: 8px;
    width: 8px;
    background-color: #00ff00; 
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 5px #00ff00;
}

/* --- Chat Bubble Customization --- */
.message.bot {
    background-color: #E8F5E9 !important; /* Light Green background for AI */
    border: 1px solid #C8E6C9 !important;
    color: #1b5e20 !important; /* Dark Green text */
}
"""

# --- HTML Header Structure ---
# This string is injected directly into the Gradio interface using gr.HTML()
HEADER_HTML: str = f"""
<div class='moi-header'>
    
    <div class='header-title-group' dir='rtl'>
        <img src='{LOGO_PATH}' class='moi-logo' alt='MOI Logo'>
        <div class='header-text'>
            <h1>المساعد الذكي الشامل</h1>
            <p>MOI Universal Assistant</p>
        </div>
    </div>
    
    <div class='tech-badges'>
        <span class='badge'><span class='dot'></span> System Online</span>
        <span class='badge'>🤖 Model: ALLaM-7B-Instruct</span>
        <span class='badge'>⚡ RAG: Hybrid (Dense+BM25)</span>
        <span class='badge'>v2.1.0</span>
    </div>
</div>
"""