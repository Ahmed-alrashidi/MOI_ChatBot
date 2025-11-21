import gradio as gr
import time
from typing import Optional, Tuple, List, Any
from core.model_loader import ModelManager
from ui.theme import MOI_CSS, HEADER_HTML
from utils.logger import setup_logger

# Initialize UI Logger
logger = setup_logger(__name__)

def create_app(rag_chain: Any) -> gr.Blocks:
    """
    Builds and returns the Gradio Blocks app.
    Receives the initialized 'rag_chain' from main.py.
    """
    
    # --- Logic Handlers ---
    
    def start_chat(lang: str) -> Tuple[Any, ...]:
        """Transitions from Welcome Screen to Chat Screen based on language."""
        logger.info(f"🌐 User selected language: {lang}")
        
        if lang == "Arabic":
            return (
                gr.update(visible=False), # Hide Welcome
                gr.update(visible=True),  # Show Chat
                [(None, "👋 حياك الله! أنا مساعدك الذكي لخدمات وزارة الداخلية. تفضل بطرح سؤالك.")],
                lang,
                gr.update(label="المحادثة الفورية", rtl=True),
                gr.update(placeholder="اكتب سؤالك هنا...", rtl=True)
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                [(None, "👋 Hello! I am your MOI Smart Assistant. How can I help you today?")],
                lang,
                gr.update(label="Live Chat", rtl=False),
                gr.update(placeholder="Type your question here...", rtl=False)
            )

    def chat_response(message: str, history: List[Tuple], audio_file: Optional[str], lang_val: str) -> List[Tuple]:
        """Handles user input (Text/Audio) and retrieves response from RAG Chain."""
        
        user_display = message
        
        # 1. Handle Audio Input (Priority)
        if audio_file:
            logger.info("🎤 Audio input detected. Processing with Whisper...")
            try:
                asr_pipe = ModelManager.get_asr_pipeline()
                if asr_pipe:
                    target_lang = "ar" if lang_val == "Arabic" else "en"
                    # Transcribe
                    out = asr_pipe(audio_file, generate_kwargs={"language": target_lang})
                    text = out["text"].strip()
                    
                    message = text
                    user_display = f"🎤 {text}"
                    logger.info(f"📝 Transcribed text: {text}")
                else:
                    logger.error("❌ Whisper model not loaded.")
                    return history + [[None, "⚠️ Error: Voice recognition model is not ready."]]
            except Exception as e:
                logger.error(f"❌ Audio processing error: {e}")
                return history + [[None, f"❌ Audio Error: {str(e)}"]]

        # 2. Validate Empty Input
        if not message or not message.strip():
            return history

        # 3. Generate Response via RAG
        if not rag_chain:
            logger.critical("❌ RAG Chain is NOT initialized.")
            response = "⚠️ System Error: The AI Brain is not loaded. Please contact administrator."
        else:
            try:
                logger.info(f"🤖 Sending query to RAG: {message[:50]}...")
                response = rag_chain.answer(message)
            except Exception as e:
                logger.error(f"❌ RAG Inference failed: {e}")
                response = f"❌ عذراً، حدث خطأ في النظام: {str(e)}"

        # 4. Update History
        history.append((user_display, response))
        return history

    def reset_app():
        """Resets the session to the language selection screen."""
        logger.info("🔄 User reset the application.")
        return (
            gr.update(visible=True),  # Show Welcome
            gr.update(visible=False), # Hide Chat
            None,                     # Clear State
            []                        # Clear History
        )

    def clear_inputs(): 
        return "", None

    # --- UI Layout Construction ---
    with gr.Blocks(theme=gr.themes.Soft(), css=MOI_CSS, title="MOI Assistant") as demo:
        
        # Session State
        lang_state = gr.State(value="Arabic")

        # Header
        gr.HTML(HEADER_HTML)

        # SCREEN 1: Welcome & Language Selection
        with gr.Group(visible=True) as welcome_screen:
            gr.Markdown("<h3 style='text-align: center;'>🌍 Please select your preferred language</h3>")
            with gr.Row():
                btn_ar = gr.Button("العربية 🇸🇦", variant="primary", elem_classes=["lang-btn"])
                btn_en = gr.Button("English 🇬🇧", variant="secondary", elem_classes=["lang-btn"])

        # SCREEN 2: Chat Interface
        with gr.Group(visible=False) as chat_screen:
            with gr.Row():
                # Left Column: Chatbot
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="Chat", height=500)
                    with gr.Row():
                        msg = gr.Textbox(show_label=False, container=False, scale=4, placeholder="Type here...")
                        submit_btn = gr.Button("🚀", variant="primary", scale=1)

                # Right Column: Tools
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ أدوات / Tools")
                    
                    # Note: type="filepath" is important for Whisper processing
                    audio_input = gr.Audio(source="microphone", type="filepath", label="Voice Input")
                    
                    gr.Markdown("---")
                    restart_btn = gr.Button("🔄 Change Language", variant="secondary")

        # --- Event Wiring ---
        
        # Language Selection
        btn_ar.click(fn=lambda: start_chat("Arabic"), outputs=[welcome_screen, chat_screen, chatbot, lang_state, chatbot, msg])
        btn_en.click(fn=lambda: start_chat("English"), outputs=[welcome_screen, chat_screen, chatbot, lang_state, chatbot, msg])

        # Chat Submission (Enter Key)
        msg.submit(chat_response, [msg, chatbot, audio_input, lang_state], [chatbot]) \
           .then(clear_inputs, None, [msg, audio_input])
        
        # Chat Submission (Button Click)
        submit_btn.click(chat_response, [msg, chatbot, audio_input, lang_state], [chatbot]) \
                  .then(clear_inputs, None, [msg, audio_input])
        
        # Restart Button
        restart_btn.click(reset_app, outputs=[welcome_screen, chat_screen, lang_state, chatbot])

    return demo