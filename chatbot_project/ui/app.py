# =========================================================================
# File Name: ui/app.py
# Project: Absher Smart Assistant (MOI ChatBot)
# Architecture: Cross-Lingual Hybrid RAG (BGE-M3 + BM25 + ALLaM-7B)
#
# Affiliation: King Abdullah University of Science and Technology (KAUST)
# Team: Ahmed AlRashidi, Sultan Alshaibani, Fahad Alqahtani, 
#       Rakan Alharbi, Sultan Alotaibi, Abdulaziz Almutairi.
# Advisors: Prof. Naeemullah Khan & Dr. Salman Khan
# =========================================================================

import gradio as gr
import time
from typing import List, Tuple, Optional, Any

from config import Config
from core.model_loader import ModelManager
from ui.theme import MOI_CSS, HEADER_HTML, MOI_THEME
from utils.logger import setup_logger
from utils.tts import generate_speech

# Initialize UI Logger
logger = setup_logger(__name__)

def create_app(rag_system: Any) -> gr.Blocks:
    """
    Constructs the Gradio UI with A100 optimizations.
    Connects the UI to the RAG Pipeline and ASR/TTS modules.
    """
    
    # --- Interaction Logic ---
    def chat_pipeline(
        user_text: str, 
        history: List[Tuple[str, str]], 
        audio_path: Optional[str]
    ):
        """
        The Master Handler: Audio -> Text -> RAG -> Response -> TTS
        """
        if not user_text and not audio_path:
            return history, None, "", None

        # 1. Transcribe Audio (if present)
        if audio_path:
            logger.info("🎤 Processing Audio Input...")
            try:
                asr_pipe = ModelManager.get_asr_pipeline()
                if asr_pipe:
                    # Whisper Inference
                    out = asr_pipe(audio_path)
                    transcribed = out["text"].strip()
                    if transcribed:
                        user_text = transcribed  # Override text with transcription
                else:
                    logger.warning("⚠️ ASR Model not loaded.")
            except Exception as e:
                logger.error(f"❌ ASR Error: {e}")

        # 2. Update UI immediately (User Message)
        # We append a placeholder for the bot response
        history = history + [[user_text, None]]
        yield history, None, "", None # Yield 1: Show user message

        # 3. RAG Generation (The "Thinking" Phase)
        start_time = time.time()
        try:
            # Pass history excluding the current None placeholder
            context_history = [tuple(h) for h in history[:-1]]
            
            response_text = rag_system.run(user_text, context_history)
            
            # Calculate latency for logs
            latency = time.time() - start_time
            logger.info(f"✅ Response generated in {latency:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ RAG Error: {e}")
            response_text = "عذراً، واجهت مشكلة تقنية في معالجة طلبك. يرجى المحاولة مرة أخرى."

        # 4. Update UI (Bot Message)
        history[-1][1] = response_text
        yield history, None, "", None # Yield 2: Show text response

        # 5. Generate TTS (The "Speaking" Phase)
        audio_output = None
        try:
            audio_output = generate_speech(response_text)
        except Exception as e:
            logger.error(f"❌ TTS Error: {e}")

        # Final Yield: Add Audio
        yield history, audio_output, "", None

    def clear_context():
        """Resets the interface and clears memory"""
        return [], None, "", None

    # --- UI Layout Construction ---
    with gr.Blocks(theme=MOI_THEME, css=MOI_CSS, title="Absher Smart Assistant") as demo:
        
        # 1. Header
        gr.HTML(HEADER_HTML)

        # 2. Main Layout
        with gr.Row():
            
            # Left Column: Chat Window
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="المحادثة",
                    show_label=False,
                    bubble_full_width=False,
                    avatar_images=(None, "ui/assets/moi_logo.png"), # Ensure logo exists or use None
                    height=550,
                    rtl=True,
                    elem_classes=["chatbot-container"]
                )
                
                # Audio Player (Hidden initially, appears when TTS is ready)
                tts_player = gr.Audio(
                    label="الرد الصوتي", 
                    autoplay=True, 
                    visible=True, 
                    elem_id="tts-player",
                    interactive=False
                )

            # Right Column: Controls
            with gr.Column(scale=1, elem_classes=["controls-panel"]):
                gr.Markdown("### 🗨️ ابدأ المحادثة")
                
                msg_input = gr.Textbox(
                    placeholder="كتب استفسارك هنا أو استخدم الميكروفون...",
                    label="الاستفسار النصي",
                    lines=2,
                    max_lines=4,
                    rtl=True,
                    elem_id="msg-input"
                )
                
                audio_input = gr.Audio(
                    source="microphone", 
                    type="filepath", 
                    label="تسجيل صوتي",
                    elem_id="audio-input"
                )
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ مسح", variant="secondary")
                    submit_btn = gr.Button("🚀 إرسال", variant="primary")

                # Examples Section
                gr.Markdown("### 💡 أسئلة شائعة")
                gr.Examples(
                    examples=[
                        ["كيف أجدد رخصة القيادة؟"],
                        ["ما هي شروط إصدار الإقامة؟"],
                        ["كم رسوم تجديد الجواز؟"],
                        ["كيف أبلغ عن حادث بسيط؟"]
                    ],
                    inputs=msg_input,
                    label="نماذج"
                )

        # --- Event Wiring ---
        # 1. Submit via Button
        submit_btn.click(
            chat_pipeline,
            inputs=[msg_input, chatbot, audio_input],
            outputs=[chatbot, tts_player, msg_input, audio_input]
        )

        # 2. Submit via Enter Key
        msg_input.submit(
            chat_pipeline,
            inputs=[msg_input, chatbot, audio_input],
            outputs=[chatbot, tts_player, msg_input, audio_input]
        )

        # 3. Clear Button
        clear_btn.click(
            clear_context,
            outputs=[chatbot, tts_player, msg_input, audio_input]
        )

    return demo