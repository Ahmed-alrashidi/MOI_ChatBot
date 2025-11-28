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
    Constructs the Gradio User Interface for the MOI Universal Assistant.
    
    Features:
    - Text and Audio Input (Multimodal).
    - Real-time RAG inference.
    - Text-to-Speech (TTS) output.
    - RTL support for Arabic visualization.
    
    Args:
        rag_chain: The initialized ProRAGChain instance to handle logic.
        
    Returns:
        gr.Blocks: The compiled Gradio application ready to launch.
    """
    
    # --- Logic Handlers ---
    
    def chat_response(message: str, history: List[Tuple[Optional[str], Optional[str]]], audio_file: Optional[str]) -> Tuple[List[Tuple], Optional[str]]:
        """
        The central processing function for the chat interface.
        
        Pipeline:
        1. Audio Transcription (if microphone used).
        2. Input Validation.
        3. RAG Inference (Retrieval + Generation).
        4. TTS Generation (Text-to-Speech).
        """
        user_display = message
        
        # 1. Handle Audio Input (Speech-to-Text)
        if audio_file:
            logger.info("🎤 Audio input detected. Processing with Whisper...")
            try:
                asr_pipe = ModelManager.get_asr_pipeline()
                if asr_pipe:
                    # Transcribe audio file
                    out = asr_pipe(audio_file)
                    text = out["text"].strip()
                    
                    # Update message and display variable
                    message = text
                    user_display = f"🎤 {text}" # Add icon to indicate voice input
                    logger.info(f"📝 Transcribed text: {text}")
                else:
                    logger.error("❌ Whisper model not loaded.")
            except Exception as e:
                logger.error(f"❌ Audio processing error: {e}")

        # 2. Validate Input (Prevent empty queries)
        if not message or not message.strip():
            return history, None

        # 3. Generate Response (RAG)
        if not rag_chain:
            response = "⚠️ System Error: AI Brain not loaded."
        else:
            try:
                # Core RAG Logic
                response = rag_chain.answer(message, history=history)
            except Exception as e:
                logger.error(f"❌ RAG Inference failed: {e}")
                response = f"❌ Error: {str(e)}"

        # 4. Update Chat History
        history.append((user_display, response))
        
        # 5. Generate TTS (Text-to-Speech)
        # CRITICAL: Strip HTML tags added by the RAG pipeline for UI formatting.
        # The TTS engine reads raw text, so tags like <div> must be removed.
        clean_text = response.replace("<div dir='rtl' style='text-align: right;'>", "") \
                             .replace("<div dir='ltr' style='text-align: left;'>", "") \
                             .replace("</div>", "")
        
        audio_path = generate_speech(clean_text)
        
        return history, audio_path

    # --- Helper Functions for UI Actions ---
    
    def clean_after_send():
        """Resets input fields (Text & Audio) after sending, but keeps history."""
        return "", None

    def full_reset():
        """Performs a hard reset: Clears Chat, Inputs, and TTS Player."""
        return [], "", None, None

    # --- UI Layout Construction ---
    with gr.Blocks(theme=gr.themes.Soft(), css=MOI_CSS, title="MOI Universal Assistant") as demo:
        
        # 1. Header Section
        gr.HTML(HEADER_HTML)

        # 2. Main Chat Area
        with gr.Group():
            # Chatbot component with RTL enabled for Arabic
            chatbot = gr.Chatbot(label="MOI Smart Assistant", height=500, rtl=True)
            
            # Audio Player for the AI's response (Autoplay disabled for better UX)
            tts_player = gr.Audio(
                label="🔊 قراءة الإجابة / Read Response", 
                autoplay=False, 
                visible=True, 
                type="filepath"
            )

            # Input Area (Text & Send Button)
            with gr.Row():
                msg = gr.Textbox(
                    show_label=False, 
                    container=False, 
                    scale=4, 
                    placeholder="تفضل بطرح سؤالك بأي لغة... / Ask here...", 
                    autofocus=True,
                    lines=1
                )
                submit_btn = gr.Button("🚀 إرسال / Send", variant="primary", scale=1)

            # Secondary Inputs (Mic & Clear)
            with gr.Row():
                with gr.Column(scale=1):
                    # Microphone Input
                    audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Voice Input")
                
                with gr.Column(scale=0.2):
                    clear_btn = gr.Button("🗑️ مسح / Clear", variant="secondary")

        # --- Event Wiring ---

        # Case A: Submit via Enter Key in Textbox
        msg.submit(chat_response, [msg, chatbot, audio_input], [chatbot, tts_player]) \
           .then(clean_after_send, None, [msg, audio_input])
        
        # Case B: Submit via 'Send' Button
        submit_btn.click(chat_response, [msg, chatbot, audio_input], [chatbot, tts_player]) \
                  .then(clean_after_send, None, [msg, audio_input])
        
        # Case C: Clear Conversation
        clear_btn.click(full_reset, None, [chatbot, msg, audio_input, tts_player])

    return demo