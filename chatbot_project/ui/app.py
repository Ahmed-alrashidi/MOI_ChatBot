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
    Updated for Universal Polyglot Support & Context Awareness.
    """
    
    # --- Logic Handlers ---
    
    def chat_response(message: str, history: List[Tuple], audio_file: Optional[str]) -> List[Tuple]:
        """
        Handles user input without pre-set language.
        Whisper will auto-detect language for audio.
        RAG Chain will auto-detect language for text.
        Passes history to RAG chain for contextual understanding.
        """
        user_display = message
        
        # 1. Handle Audio Input (Auto-Detect Language)
        if audio_file:
            logger.info("🎤 Audio input detected. Processing with Whisper (Auto-Detect)...")
            try:
                asr_pipe = ModelManager.get_asr_pipeline()
                if asr_pipe:
                    # Whisper detects language automatically
                    out = asr_pipe(audio_file)
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

        # 3. Generate Response via RAG (Language Agnostic + Context Aware)
        if not rag_chain:
            logger.critical("❌ RAG Chain is NOT initialized.")
            response = "⚠️ System Error: The AI Brain is not loaded. Please contact administrator."
        else:
            try:
                logger.info(f"🤖 Sending query to RAG: {message[:50]}...")
                
                # 🔥 CRITICAL UPDATE: Passing 'history' allows the RAG chain
                # to understand follow-up questions like "How much does it cost?"
                response = rag_chain.answer(message, history=history)
                
            except Exception as e:
                logger.error(f"❌ RAG Inference failed: {e}")
                response = f"❌ Error: {str(e)}"

        # 4. Update History
        history.append((user_display, response))
        return history

    def clear_inputs(): 
        return "", None

    # --- UI Layout Construction ---
    with gr.Blocks(theme=gr.themes.Soft(), css=MOI_CSS, title="MOI Universal Assistant") as demo:
        
        # Header
        gr.HTML(HEADER_HTML)

        # Main Interface (Single Screen)
        with gr.Group():
            # Chatbot Window
            # RTL=True favors Arabic alignment but works for mixed content too
            chatbot = gr.Chatbot(label="MOI Smart Assistant", height=600, rtl=True)
            
            with gr.Row():
                # Text Input
                msg = gr.Textbox(
                    show_label=False, 
                    container=False, 
                    scale=4, 
                    placeholder="تفضل بطرح سؤالك بأي لغة... / Ask here in any language..."
                )
                submit_btn = gr.Button("🚀 إرسال / Send", variant="primary", scale=1)

            with gr.Row():
                # Tools Row
                with gr.Column(scale=1):
                    # Audio Input (Auto-Detect)
                    audio_input = gr.Audio(source="microphone", type="filepath", label="🎙️ Voice Input (Auto-Detect)")
                
                with gr.Column(scale=0.2):
                    # Clear Button
                    clear_btn = gr.Button("🗑️ مسح / Clear", variant="secondary")

        # --- Event Wiring ---

        # Chat Submission (Enter Key)
        msg.submit(chat_response, [msg, chatbot, audio_input], [chatbot]) \
           .then(clear_inputs, None, [msg, audio_input])
        
        # Chat Submission (Button Click)
        submit_btn.click(chat_response, [msg, chatbot, audio_input], [chatbot]) \
                  .then(clear_inputs, None, [msg, audio_input])
        
        # Clear Button Logic
        clear_btn.click(lambda: ([], "", None), None, [chatbot, msg, audio_input])

    return demo