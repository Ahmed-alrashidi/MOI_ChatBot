# MOI Theme Colors and Custom CSS

MOI_CSS = """
.gradio-container { background-color: #f4f6f8 !important; }
.moi-header {
    text-align: center;
    padding: 30px;
    background: linear-gradient(90deg, #006C35 0%, #004D26 100%);
    border-radius: 10px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.moi-header h1 { color: white !important; font-size: 2.5em; margin: 0; }
.moi-header p { color: #d4af37 !important; font-size: 1.2em; margin-top: 5px; }
.lang-btn { font-size: 1.2em; height: 60px; border-radius: 8px !important; }
"""

HEADER_HTML = """
<div class='moi-header'>
    <h1>المساعد الذكي لوزارة الداخلية</h1>
    <p>MOI Smart Assistant | Powered by ALLaM-7B</p>
</div>
"""