import gradio as gr
from typing import Optional, Tuple, List, Any
from core.model_loader import ModelManager
from ui.theme import MOI_CSS, HEADER_HTML
from utils.logger import setup_logger
from utils.tts import generate_speech

# Initialize UI Logger
logger = setup_logger(__name__)

def create_app(rag_chain: Any) -> gr.Blocks:
    """
    Builds and returns the Gradio Blocks app.
    Final Version: Tech Theme + Polyglot + TTS (Manual Play) + Bug Fixes.
    """
    
    # --- Logic Handlers ---
    
    def chat_response(message: str, history: List[Tuple], audio_file: Optional[str]) -> Tuple[List[Tuple], Optional[str]]:
        user_display = message
        
        # 1. Handle Audio Input
        if audio_file:
            logger.info("🎤 Audio input detected. Processing with Whisper...")
            try:
                asr_pipe = ModelManager.get_asr_pipeline()
                if asr_pipe:
                    out = asr_pipe(audio_file)
                    text = out["text"].strip()
                    message = text
                    user_display = f"🎤 {text}"
                    logger.info(f"📝 Transcribed text: {text}")
                else:
                    logger.error("❌ Whisper model not loaded.")
            except Exception as e:
                logger.error(f"❌ Audio processing error: {e}")

        # 2. Validate Input
        if not message or not message.strip():
            return history, None

        # 3. Generate Response
        if not rag_chain:
            response = "⚠️ System Error: AI Brain not loaded."
        else:
            try:
                response = rag_chain.answer(message, history=history)
            except Exception as e:
                logger.error(f"❌ RAG Inference failed: {e}")
                response = f"❌ Error: {str(e)}"

        # 4. Update History
        history.append((user_display, response))
        
        # 5. Generate TTS
        # Clean HTML tags for speech
        clean_text = response.replace("<div dir='rtl' style='text-align: right;'>", "") \
                             .replace("<div dir='ltr' style='text-align: left;'>", "") \
                             .replace("</div>", "")
        
        audio_path = generate_speech(clean_text)
        
        return history, audio_path

    # --- Helper Functions for UI Actions ---
    
    def clean_after_send():
        """Clears input box and audio input only (Keep chat history)."""
        return "", None

    def full_reset():
        """Clears EVERYTHING: Chat, Input, Audio Input, and TTS Player."""
        return [], "", None, None

    # --- UI Layout ---
    with gr.Blocks(theme=gr.themes.Soft(), css=MOI_CSS, title="MOI Universal Assistant") as demo:
        
        gr.HTML(HEADER_HTML)

        with gr.Group():
            chatbot = gr.Chatbot(label="MOI Smart Assistant", height=500, rtl=True)
            
            # TTS Player
            # ✅ تم التحديث: تعطيل التشغيل التلقائي (autoplay=False)
            tts_player = gr.Audio(label="🔊 قراءة الإجابة / Read Response", autoplay=False, visible=True, type="filepath")

            with gr.Row():
                msg = gr.Textbox(
                    show_label=False, 
                    container=False, 
                    scale=4, 
                    placeholder="تفضل بطرح سؤالك بأي لغة... / Ask here...", 
                    autofocus=True
                )
                submit_btn = gr.Button("🚀 إرسال / Send", variant="primary", scale=1)

            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(source="microphone", type="filepath", label="🎙️ Voice Input")
                
                with gr.Column(scale=0.2):
                    clear_btn = gr.Button("🗑️ مسح / Clear", variant="secondary")

        # --- Event Wiring ---

        # 1. Submit via Enter Key
        msg.submit(chat_response, [msg, chatbot, audio_input], [chatbot, tts_player]) \
           .then(clean_after_send, None, [msg, audio_input])
        
        # 2. Submit via Button
        submit_btn.click(chat_response, [msg, chatbot, audio_input], [chatbot, tts_player]) \
                  .then(clean_after_send, None, [msg, audio_input])
        
        # 3. Full Reset
        clear_btn.click(full_reset, None, [chatbot, msg, audio_input, tts_player])

    return demo