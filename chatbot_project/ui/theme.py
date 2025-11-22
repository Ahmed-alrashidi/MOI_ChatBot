import os

LOGO_PATH = "file/ui/moi_logo.png"

MOI_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');

body, .gradio-container { 
    font-family: 'Tajawal', sans-serif !important; 
    background-color: #f0f2f5 !important; 
}

/* --- Tech Header Styling --- */
.moi-header {
    display: flex;
    /* FIX 1: Ensure main elements are spaced across the bar */
    justify-content: space-between; 
    align-items: center;
    padding: 15px 40px; 
    background: linear-gradient(135deg, #004D26 0%, #006C35 100%);
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(0, 77, 38, 0.2);
    margin-bottom: 20px;
    border-bottom: 4px solid #d4af37;
}

/* FIX 2: Grouping the Logo and Title (RTL Alignment) */
.header-title-group {
    display: flex;
    flex-direction: row-reverse; 
    align-items: center;
    gap: 20px;
    text-align: right;
}

.moi-logo {
    width: 80px; /* Reduced size slightly for better fit */
    height: auto;
    filter: drop-shadow(0 4px 6px rgba(0,0,0,0.3));
}

.header-title-group h1 { 
    color: #ffffff !important; 
    font-size: 1.8em; /* Adjusted font size */
    margin: 0; 
    font-weight: 700;
}

.header-title-group p { 
    color: #d4af37 !important; 
    font-size: 0.9em; /* Adjusted font size */
    margin: 0; 
    opacity: 0.9;
}

/* --- Tech Badges (FIX 3: Left Alignment and Horizontal Flow) --- */
.tech-badges {
    display: flex;
    gap: 10px;
    /* We align these left logically, even though the whole div is flexed LTR */
    align-items: center; 
}

.badge {
    background: rgba(255, 255, 255, 0.15);
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

.dot {
    height: 8px;
    width: 8px;
    background-color: #00ff00; 
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 5px #00ff00;
}

/* --- Chat & Audio Styling --- */
.message.bot {
    background-color: #E8F5E9 !important;
    border: 1px solid #C8E6C9 !important;
    color: #1b5e20 !important;
}
"""

# HTML with separated content groups
HEADER_HTML = f"""
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