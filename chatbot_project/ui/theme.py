import os

# --- Path Configuration ---
# Correct path pointing to: ui/assets/moi_logo.png
# When running main.py from root, Gradio serves files relative to root via "file/" prefix
LOGO_PATH = "file/ui/assets/moi_logo.png"

MOI_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');

/* --- Global Reset & Font --- */
body, .gradio-container { 
    font-family: 'Tajawal', sans-serif !important; 
    background-color: #f4f6f8 !important; 
}

/* --- The Main Header Card --- */
.moi-header {
    display: flex;
    justify-content: space-between; 
    align-items: center;
    padding: 1.2rem 2.5rem; 
    background: linear-gradient(135deg, #1A5D3A 0%, #0F4026 100%); /* Official MOI Green */
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(26, 93, 58, 0.25);
    margin-bottom: 25px;
    border-bottom: 3px solid #D4AF37; /* Gold Trim */
    border-top: 1px solid rgba(255,255,255,0.1);
    position: relative;
    overflow: hidden;
}

/* Subtle background pattern overlay */
.moi-header::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: radial-gradient(circle at 20% 50%, rgba(255,255,255,0.05) 0%, transparent 25%);
    pointer-events: none;
}

/* --- Right Side: Title & Logo (RTL) --- */
.header-title-group {
    display: flex;
    flex-direction: row-reverse; 
    align-items: center;
    gap: 1.5rem;
    text-align: right;
    z-index: 2;
}

.moi-logo {
    width: 85px; 
    height: auto;
    filter: drop-shadow(0 4px 6px rgba(0,0,0,0.2));
    transition: transform 0.3s ease;
}

.moi-logo:hover {
    transform: scale(1.05);
}

.header-text h1 {
    color: #ffffff;
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0;
    line-height: 1.2;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.header-text p {
    color: #e0e0e0;
    font-size: 0.95rem;
    margin: 0;
    opacity: 0.9;
    font-weight: 500;
}

/* --- Left Side: Tech Badges (LTR Flow) --- */
.tech-badges {
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
    z-index: 2;
}

.badge {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(5px);
    padding: 6px 14px;
    border-radius: 50px;
    color: #fff;
    font-size: 0.85rem;
    font-weight: 600;
    font-family: 'Segoe UI', monospace;
    border: 1px solid rgba(255, 255, 255, 0.15);
    display: flex;
    align-items: center;
    gap: 8px;
    transition: all 0.3s ease;
}

.badge:hover {
    background: rgba(255, 255, 255, 0.2);
    transform: translateY(-2px);
    border-color: #D4AF37;
}

/* Hardware Accelerator Badge style */
.badge.gpu {
    background: linear-gradient(90deg, rgba(118,185,0,0.2) 0%, rgba(26,93,58,0.4) 100%);
    border: 1px solid #76b900;
    color: #e8f5e9;
}

/* Live Indicator Dot with Pulse */
.dot {
    height: 8px;
    width: 8px;
    background-color: #00ff88; 
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 8px #00ff88;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.7); }
    70% { box-shadow: 0 0 0 6px rgba(0, 255, 136, 0); }
    100% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0); }
}

/* --- Chat Interface Styling --- */
.message-row {
    margin-bottom: 12px;
}

/* User Message */
.message.user {
    background-color: #ffffff !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 12px 12px 2px 12px !important; /* Point to right */
    color: #333 !important;
    font-weight: 500;
}

/* Bot Message */
.message.bot {
    background-color: #F1F8E9 !important; /* Light Green Tint */
    border: 1px solid #C5E1A5 !important;
    border-radius: 12px 12px 12px 2px !important; /* Point to left */
    color: #004D26 !important;
    line-height: 1.6;
}

/* RTL Support inside chat */
.message {
    direction: rtl;
    text-align: right;
}
/* If text is detected as English via a class (optional), flip it */
.message.ltr {
    direction: ltr;
    text-align: left;
}

/* Footer/Disclaimer */
footer {
    display: none !important; /* Hide default Gradio footer */
}
"""

# HTML Component
HEADER_HTML = f"""
<div class='moi-header'>
    <div class='tech-badges'>
        <div class='badge gpu'>
            <span>🚀 NVIDIA A100</span>
        </div>
        <div class='badge'>
            <span>🤖 ALLaM-7B</span>
        </div>
        <div class='badge'>
            <span>🧠 RAG v4.0</span>
        </div>
        <div class='badge'>
            <span class='dot'></span>
            <span>System Online</span>
        </div>
    </div>

    <div class='header-title-group'>
        <div class='header-text'>
            <h1>المساعد الذكي لوزارة الداخلية</h1>
            <p>MOI Smart Assistant • Powered by Generative AI</p>
        </div>
        <img src='{LOGO_PATH}' class='moi-logo' alt='MOI Logo'>
    </div>
</div>
"""