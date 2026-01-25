# =========================================================================
# File Name: ui/theme.py
# Purpose: Visual Identity, Styling, and Branding Configuration.
# Project: Absher Smart Assistant (MOI ChatBot)
# Features:
# - Dynamic Theming: Persistent Dark/Light mode logic via JavaScript.
# - Brand Consistency: Custom CSS for Saudi MOI colors (Emerald & Gold).
# - Typography: Integrated 'Tajawal' Google Font for high-quality Arabic rendering.
# - Responsive Components: Custom badges and animated status indicators.
# =========================================================================

import gradio as gr

# --- 1. JAVASCRIPT LOGIC (Client-Side) ---

# THEME_JS: Handles initial theme loading based on user history or system time.
# Logic: If no saved preference, automatically enables Dark Mode between 6 PM and 6 AM.
THEME_JS = """
function() {
    const savedTheme = localStorage.getItem('moi_theme');
    if (savedTheme) {
        if (savedTheme === 'dark') {
            document.body.classList.add('dark');
        }
    } else {
        const hour = new Date().getHours();
        const isNight = hour < 6 || hour >= 18;
        if (isNight) {
            document.body.classList.add('dark');
        }
    }
    return [];
}
"""

# TOGGLE_JS: Manages the manual switch between Light and Dark modes.
# Saves the user's preference to 'localStorage' for persistence across sessions.
TOGGLE_JS = """
function() {
    document.body.classList.toggle('dark');
    const isDark = document.body.classList.contains('dark');
    localStorage.setItem('moi_theme', isDark ? 'dark' : 'light');
}
"""

# --- 2. CUSTOM CSS (UX & Branding) ---

# MOI_CSS: Overrides Gradio default styles with a bespoke identity.
# It defines CSS variables (:root) to allow seamless theme switching.
MOI_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;800&display=swap');

:root {
    --moi-bg: #F9FAFB;
    --moi-card-bg: #FFFFFF;
    --moi-text-main: #111827;
    --moi-text-sub: #4B5563;
    --moi-green-primary: #114b2a;
    --moi-gold: #C5A059;
    --moi-border: #E5E7EB;
    --bot-bubble: #F3F4F6;
    --user-bubble: #114b2a;
    --user-text: #FFFFFF;
}

/* Dark Mode Color Overrides */
body.dark {
    --moi-bg: #0d1117;
    --moi-card-bg: #161b22;
    --moi-text-main: #e6edf3;
    --moi-text-sub: #8b949e;
    --moi-green-primary: #238636;
    --moi-gold: #D2B48C;
    --moi-border: #30363d;
    --bot-bubble: #21262d;
    --user-bubble: #238636;
    --user-text: #FFFFFF;
}

/* Global Font & Animation Smoothing */
body, .gradio-container, button, input, textarea { 
    font-family: 'Tajawal', sans-serif !important; 
    background-color: var(--moi-bg) !important; 
    color: var(--moi-text-main) !important;
    transition: all 0.2s ease-in-out;
}

/* Header Component Styling */
.moi-header {
    background: linear-gradient(135deg, #0d321d 0%, #114b2a 100%);
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0 10px 25px -5px rgba(17, 75, 42, 0.4);
    border-bottom: 3px solid var(--moi-gold);
    color: white;
    margin-bottom: 20px;
}

.title-section h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

/* Animated Badges for GPU and Model status */
.badge {
    background: rgba(255, 255, 255, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 6px 14px;
    border-radius: 99px;
    font-size: 0.85rem;
    color: white;
    display: flex;
    align-items: center;
    gap: 8px;
}

.badge.gpu span.dot {
    height: 8px; width: 8px; 
    background-color: #4ade80; 
    border-radius: 50%;
    box-shadow: 0 0 12px #4ade80;
    animation: pulse 2s infinite; /* Pulsing effect for active GPU indicator */
}
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }

/* Chatbot Interface Polishing */
.chatbot-container {
    border: 1px solid var(--moi-border) !important;
    border-radius: 16px !important;
    background: var(--moi-card-bg) !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

/* Distinct Bubble Styles for User vs. AI */
.user-message {
    background-color: var(--user-bubble) !important;
    color: var(--user-text) !important;
    border-radius: 18px !important;
    padding: 12px 18px !important;
}

.bot-message {
    background-color: var(--bot-bubble) !important;
    color: var(--moi-text-main) !important;
    border-radius: 18px !important;
    border: 1px solid var(--moi-border);
    padding: 16px !important;
}

/* Interactive Element Styling */
.gradio-container input, .gradio-container textarea {
    background-color: var(--moi-card-bg) !important;
    border: 1px solid var(--moi-border) !important;
    border-radius: 8px !important;
}
.gradio-container input:focus, .gradio-container textarea:focus {
    border-color: var(--moi-green-primary) !important;
    box-shadow: 0 0 0 2px rgba(17, 75, 42, 0.2) !important;
}

/* UI Cleanup: Removing Gradio default footer */
footer { display: none !important; }
"""

# --- 3. HTML HEADER (Branding Asset) ---

# This HTML block is injected into the top of the app to provide 
# official branding, including the Saudi emblem and project description.
HEADER_HTML = """
<div class='moi-header'>
    <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:20px;'>
        <div class='title-section' style='text-align: right;'>
            <div style='display:flex; align-items:center; gap:15px;'>
                <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Emblem_of_Saudi_Arabia_%282%29.svg/150px-Emblem_of_Saudi_Arabia_%282%29.svg.png' style='height:60px; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));'>
                <div>
                    <h1>مساعد أبشر الذكي</h1>
                    <p style='font-size:0.95rem; opacity:0.9; margin-top:4px;'>الذكاء الاصطناعي السيادي | Sovereign AI</p>
                </div>
            </div>
        </div>
        <div style='display:flex; gap:12px; flex-wrap:wrap;'>
            <div class='badge gpu'>
                <span class='dot'></span>
                <span>NVIDIA A100</span>
            </div>
            <div class='badge'>
                <span>🇸🇦 ALLaM-7B</span>
            </div>
        </div>
    </div>
</div>
"""

# --- 4. THEME OBJECT (Python-side Config) ---

# MOI_THEME: The Gradio Soft theme object.
# Configures the primary color hues and integrates the Tajawal font via Google Fonts.
MOI_THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="stone",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Tajawal"), "sans-serif"]
).set(
    body_background_fill="#F9FAFB",
    block_background_fill="#FFFFFF",
    block_border_color="#E5E7EB",
    input_background_fill="#FFFFFF",
    button_primary_background_fill="#114b2a",
    button_primary_text_color="white",
    body_background_fill_dark="#0d1117",
    block_background_fill_dark="#161b22",
    block_border_color_dark="#30363d",
    input_background_fill_dark="#0d1117",
    button_primary_background_fill_dark="#238636",
    button_primary_text_color_dark="white"
)