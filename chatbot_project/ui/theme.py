# =========================================================================
# File Name: ui/theme.py
# Project: Absher Smart Assistant (MOI ChatBot)
# Architecture: Cross-Lingual Hybrid RAG (BGE-M3 + BM25 + ALLaM-7B)
#
# Affiliation: King Abdullah University of Science and Technology (KAUST)
# Team: Ahmed AlRashidi, Sultan Alshaibani, Fahad Alqahtani, 
#       Rakan Alharbi, Sultan Alotaibi, Abdulaziz Almutairi.
# Advisors: Prof. Naeemullah Khan & Dr. Salman Khan
# =========================================================================

import gradio as gr

# --- 1. Custom CSS (Government Grade Design) ---
MOI_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');

/* Global Settings */
body, .gradio-container { 
    font-family: 'Tajawal', sans-serif !important; 
    background-color: #F3F4F6 !important; 
}

/* --- Header Card --- */
.moi-header {
    background: linear-gradient(135deg, #114b2a 0%, #0d321d 100%); /* Deep Emerald */
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 8px 20px rgba(17, 75, 42, 0.25);
    border-bottom: 4px solid #C5A059; /* Gold Trim */
    color: white;
    text-align: center;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}

/* Header Content Layout */
.header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 15px;
    position: relative;
    z-index: 2;
}

.title-section h1 {
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
    color: #fff;
    letter-spacing: -0.5px;
}
.title-section p {
    color: #e0e0e0;
    margin: 5px 0 0 0;
    font-size: 1rem;
    opacity: 0.9;
}

/* Tech Badges (Right Side) */
.tech-badges {
    display: flex;
    gap: 10px;
}

.badge {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 0.85rem;
    display: flex;
    align-items: center;
    gap: 6px;
    backdrop-filter: blur(4px);
}

.badge.gpu span.dot {
    height: 8px;
    width: 8px;
    background-color: #00ff88;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 8px #00ff88;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.1); }
    100% { opacity: 1; transform: scale(1); }
}

/* --- Chat Interface --- */
.chatbot-container {
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
    background: white !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

/* User Message Bubble */
.user-message {
    background-color: #114b2a !important; /* MOI Green */
    color: white !important;
    border-radius: 18px 18px 2px 18px !important;
    padding: 10px 15px !important;
    font-size: 1rem;
}

/* Bot Message Bubble */
.bot-message {
    background-color: #f3f4f6 !important; /* Light Gray */
    color: #1f2937 !important;
    border-radius: 18px 18px 18px 2px !important;
    border: 1px solid #e5e7eb;
    padding: 10px 15px !important;
    font-size: 1rem;
    line-height: 1.6;
}

/* RTL Support */
.message-wrap {
    direction: rtl;
}

/* Hide Default Footer */
footer { visibility: hidden; }
"""

# --- 2. HTML Header Component ---
HEADER_HTML = """
<div class='moi-header'>
    <div class='header-content'>
        <div class='title-section' style='text-align: right;'>
            <h1>مساعد أبشر الذكي</h1>
            <p>الذكاء الاصطناعي السيادي لخدمات وزارة الداخلية</p>
        </div>
        
        <div class='tech-badges'>
            <div class='badge gpu'>
                <span class='dot'></span>
                <span>NVIDIA A100 Online</span>
            </div>
            <div class='badge'>
                <span>🧠 RAG v4.0</span>
            </div>
            <div class='badge'>
                <span>🇸🇦 ALLaM-7B</span>
            </div>
        </div>
    </div>
</div>
"""

# --- 3. Gradio Theme Object ---
# Using Soft theme as base, customized with MOI colors
MOI_THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="stone",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Tajawal"), "sans-serif"]
).set(
    button_primary_background_fill="#114b2a",
    button_primary_background_fill_hover="#0d321d",
    button_primary_text_color="white",
    block_title_text_color="#114b2a",
    block_label_text_color="#114b2a",
    input_background_fill="white"
)