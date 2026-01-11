import gradio as gr
import gc
import torch
from typing import List, Tuple, Optional, Any

from config import Config
from core.model_loader import ModelManager
from ui.theme import MOI_CSS, HEADER_HTML
from utils.logger import setup_logger
from utils.tts import generate_speech

# Initialize UI Logger
logger = setup_logger(__name__)

def create_app(rag_chain: Any) -> gr.Blocks:
    """
    Builds the Gradio Application with A100 optimizations.
    Integrates: RAG v4.0, Whisper ASR, and MOI Theme.
    """
    
    # --- Logic Handlers ---

    def process_query(message: str, history: List[Tuple[str, str]], audio_path: Optional[str]):
        """
        Main Handler: Audio -> Text -> RAG -> Response -> TTS
        Returns: [History, Audio_Out, Msg_Box, Audio_In_Reset]
        """
        user_text = message
        
        # 1. Handle Audio Input (Whisper A100)
        if audio_path:
            logger.info("🎤 Audio detected. Transcribing with Whisper...")
            try:
                asr_pipe = ModelManager.get_asr_pipeline()
                if asr_pipe:
                    # Using the optimized pipeline
                    out = asr_pipe(audio_path)
                    transcribed = out["text"].strip()
                    if transcribed:
                        user_text = transcribed
                        logger.info(f"📝 Transcribed: {user_text}")
                else:
                    logger.warning("⚠️ Whisper not loaded. Using text only.")
            except Exception as e:
                logger.error(f"❌ ASR Error: {e}")
                # Don't overwrite text if ASR fails, maybe user typed something
                if not user_text: 
                    user_text = "عذراً، لم أتمكن من سماع الصوت بوضوح."

        # Hygiene check: If no text and no audio transcription
        if not user_text.strip():
            # Return current state without changes, but clear inputs
            return history, None, "", None

        # 2. RAG Generation (The Heavy Lifting)
        # Note: rag_chain handles memory summarization internally based on 'history'
        bot_response = rag_chain.answer(user_text, history)
        
        # 3. Text-to-Speech (TTS)
        # Generate audio file for the response
        audio_out_path = generate_speech(bot_response)

        # 4. Update UI
        updated_history = history + [(user_text, bot_response)]
        
        # Return: 
        # 1. Updated Chat History
        # 2. Generated Audio Path (Response)
        # 3. Clear Text Input ("")
        # 4. Clear Audio Input (None) -> Prevents re-sending old audio
        return updated_history, audio_out_path, "", None

    def clear_session():
        """Hard reset for memory and UI."""
        # Force memory cleanup on GPU
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Return empty states for: Chatbot, TTS Player, Text Input, Audio Input
        return [], None, "", None

    # --- UI Layout Construction ---
    with gr.Blocks(css=MOI_CSS, title="MOI Smart Assistant (A100 Powered)") as demo:
        
        # 1. Header (HTML/CSS)
        gr.HTML(HEADER_HTML)
        
        # 2. Main Chat Interface
        with gr.Column(elem_classes="chat-container"):
            chatbot = gr.Chatbot(
                label="MOI Assistant",
                elem_id="moi-chatbot",
                height=550,
                show_label=False,
                rtl=True, # Native Arabic Support
                avatar_images=(None, "ui/assets/moi_logo.png"), # Bot Avatar
                show_copy_button=True
            )
            
            # TTS Player (Hidden initially, appears with response)
            tts_player = gr.Audio(
                label="🔊 قراءة الرد",
                interactive=False, 
                autoplay=False, # User choice to play
                visible=True
            )

        # 3. Input Controls
        with gr.Row(elem_classes="input-row"):
            with gr.Column(scale=4):
                msg_input = gr.Textbox(
                    show_label=False,
                    placeholder="اكتب سؤالك هنا أو استخدم الميكروفون... / Ask here...",
                    container=False,
                    lines=1,
                    max_lines=3,
                    autofocus=True,
                    rtl=True
                )
            
            with gr.Column(scale=1, min_width=100):
                submit_btn = gr.Button("🚀 إرسال", variant="primary", size="lg")

        # 4. Utility Controls (Audio & Clear)
        with gr.Accordion("أدوات إضافية (صوت / مسح)", open=False):
            with gr.Row():
                audio_input = gr.Audio(
                    source="microphone", 
                    type="filepath", 
                    label="تسجيل صوتي",
                    show_download_button=False
                )
                clear_btn = gr.Button("🗑️ مسح المحادثة", variant="secondary")

        # --- Event Wiring ---
        
        # Define Input/Output list for cleaner code
        app_inputs = [msg_input, chatbot, audio_input]
        app_outputs = [chatbot, tts_player, msg_input, audio_input] # Added audio_input to outputs to clear it

        # Enter Key on Textbox
        msg_input.submit(process_query, inputs=app_inputs, outputs=app_outputs)
        
        # Send Button Click
        submit_btn.click(process_query, inputs=app_inputs, outputs=app_outputs)
        
        # Clear Button
        clear_btn.click(
            clear_session,
            inputs=[],
            outputs=[chatbot, tts_player, msg_input, audio_input]
        )

    return demo