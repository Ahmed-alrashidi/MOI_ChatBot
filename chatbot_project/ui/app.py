# =========================================================================
# File Name: ui/app.py
# Version: 5.0 (8 Languages + RTL/LTR + Responsive + Suggestions)
# Project: Absher Smart Assistant (MOI ChatBot)
# =========================================================================

import gradio as gr
import time
from typing import List, Optional, Any

from config import Config
from core.model_loader import ModelManager
from ui.theme import (
    MOI_CSS, HEADER_HTML, MOI_THEME, THEME_JS,
    TOGGLE_JS, SIDEBAR_TOGGLE_JS, SET_DIRECTION_JS,
    SIDEBAR_OVERLAY_HTML
)
from utils.logger import setup_logger
from utils.tts import generate_speech
from utils.telemetry import log_interaction

logger = setup_logger("UI_Layer")

# All supported languages (matching ground truth benchmark)
SUPPORTED_LANGUAGES = [
    "Arabic (\u0627\u0644\u0639\u0631\u0628\u064a\u0629)",
    "English",
    "Urdu (\u0623\u0631\u062f\u0648)",
    "French (Fran\u00e7ais)",
    "Spanish (Espa\u00f1ol)",
    "German (Deutsch)",
    "Russian (\u0420\u0443\u0441\u0441\u043a\u0438\u0439)",
    "Chinese (\u4e2d\u6587)"
]

def create_app(rag_system: Any) -> gr.Blocks:

    # --- 1. TRANSITION ---
    def enter_system(selected_lang: str):
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            selected_lang
        )

    # --- 2. CHAT LOGIC ---
    def chat_pipeline(
        user_text: str,
        history: List[List[Optional[str]]],
        audio_path: Optional[str],
        selected_lang: str,
        request: gr.Request
    ):
        if not user_text and not audio_path:
            yield history, None, gr.update(), gr.update()
            return

        if history is None:
            history = []

        if audio_path:
            logger.info("\U0001f3a4 Transcribing voice input...")
            try:
                asr_pipe = ModelManager.get_asr_pipeline()
                if asr_pipe:
                    out = asr_pipe(audio_path)
                    transcribed_text = out.get("text", "").strip()
                    if transcribed_text:
                        user_text = transcribed_text
                    else:
                        history.append(["[Voice]", "\u26a0\ufe0f \u0644\u0645 \u064a\u062a\u0645 \u0627\u0644\u062a\u0639\u0631\u0641 \u0639\u0644\u0649 \u0627\u0644\u0635\u0648\u062a."])
                        yield history, None, gr.update(value=""), gr.update(value=None)
                        return
                else:
                    history.append(["[Voice]", "\u26a0\ufe0f \u0646\u0638\u0627\u0645 \u0627\u0644\u0635\u0648\u062a \u063a\u064a\u0631 \u0645\u062a\u0627\u062d \u062d\u0627\u0644\u064a\u0627\u064b."])
                    yield history, None, gr.update(value=""), gr.update(value=None)
                    return
            except Exception as e:
                logger.error(f"ASR Fail: {e}")
                history.append(["[Voice]", "\u26a0\ufe0f \u062e\u0637\u0623 \u0641\u064a \u0645\u0639\u0627\u0644\u062c\u0629 \u0627\u0644\u0635\u0648\u062a."])
                yield history, None, gr.update(value=""), gr.update(value=None)
                return

        history.append([user_text, None])
        yield history, None, gr.update(value=""), gr.update(value=None)

        start_timestamp = time.time()
        try:
            context_history = [tuple(turn) for turn in history[:-1]]
            ai_response = rag_system.run(user_text, context_history)
        except Exception as e:
            logger.error(f"RAG Fail: {e}")
            ai_response = "\u0639\u0630\u0631\u0627\u064b\u060c \u0646\u0648\u0627\u062c\u0647 \u0636\u063a\u0637\u0627\u064b \u0641\u064a \u0627\u0644\u0646\u0638\u0627\u0645 \u062d\u0627\u0644\u064a\u0627\u064b."

        history[-1][1] = ai_response

        if request:
            try:
                log_interaction(
                    username=getattr(request, "username", "guest"),
                    query=user_text,
                    response=ai_response,
                    latency=(time.time() - start_timestamp),
                    client_ip=request.client.host if request.client else "0.0.0.0",
                    user_agent=dict(request.headers).get('user-agent', 'Unknown Device')
                )
            except Exception as e:
                logger.warning(f"\u26a0\ufe0f Telemetry logging failed: {e}")

        yield history, None, gr.update(value=""), gr.update(value=None)

    def trigger_tts(history):
        if not history or not history[-1][1]:
            return None
        return generate_speech(history[-1][1])

    # --- 3. UI ---
    with gr.Blocks(theme=MOI_THEME, css=MOI_CSS, title="Absher Smart Assistant") as demo:

        gr.HTML(SIDEBAR_OVERLAY_HTML)

        # ====== STAGE 1: WELCOME ======
        with gr.Column(elem_id="welcome-container", visible=True) as welcome_container:
            gr.HTML("""
                <div style='margin-bottom:12px;'>
                    <img src='file/ui/assets/saudi_emblem.svg'
                         style='height:80px;margin:0 auto;display:block;opacity:.9;'>
                </div>
            """)
            gr.Markdown("# \u0645\u0631\u062d\u0628\u0627\u064b \u0628\u0643 \u0641\u064a \u0645\u0633\u0627\u0639\u062f \u0623\u0628\u0634\u0631 \u0627\u0644\u0630\u0643\u064a")
            gr.Markdown("### Welcome to Absher Smart Assistant")
            gr.HTML("<div style='height:1px;background:var(--border);margin:16px 40px;'></div>")
            gr.Markdown("\u0627\u0644\u0631\u062c\u0627\u0621 \u0627\u062e\u062a\u064a\u0627\u0631 \u0627\u0644\u0644\u063a\u0629 \u0644\u0644\u0645\u062a\u0627\u0628\u0639\u0629 | Please select your language")

            init_lang_radio = gr.Radio(
                choices=SUPPORTED_LANGUAGES,
                value=SUPPORTED_LANGUAGES[0],
                show_label=False,
                elem_id="init-lang-radio"
            )
            enter_btn = gr.Button("\u062f\u062e\u0648\u0644 | Enter", variant="primary", size="lg")

            gr.HTML("""
                <div style='margin-top:20px;padding-top:16px;border-top:1px solid var(--border);
                     display:flex;align-items:center;justify-content:center;gap:14px;opacity:.7;'>
                    <img src='file/ui/assets/KAUST.png' style='height:28px;' alt='KAUST'>
                    <span style='font-size:.75rem;color:var(--text-muted);'>
                        Built by <strong style='color:var(--text-secondary);'>Team PGD+</strong> \u00b7 KAUST Academy
                    </span>
                    <img src='file/ui/assets/Saudi_made.png' style='height:28px;' alt='Saudi Made'>
                </div>
            """)

        # ====== STAGE 2: MAIN ======
        with gr.Column(visible=False) as main_container:

            # Header
            with gr.Row(elem_id="header-container"):
                with gr.Column(scale=8):
                    gr.HTML(HEADER_HTML)
                with gr.Column(scale=2, min_width=100):
                    with gr.Row():
                        settings_btn = gr.Button("\u2699\ufe0f", variant="secondary", elem_classes=["header-btn"])
                        theme_btn = gr.Button("\U0001f317", variant="secondary", elem_classes=["header-btn"])

            # Chat Workspace
            with gr.Column(elem_id="chat-workspace"):

                chatbot = gr.Chatbot(
                    show_label=False,
                    avatar_images=(None, "ui/assets/moi_logo.png"),
                    height=480,
                    rtl=True,
                    elem_classes=["chatbot-container"],
                    show_copy_button=True
                )

                # Quick Suggestions
                with gr.Row(elem_classes=["suggestion-row"]):
                    chip1 = gr.Button("\U0001f4b3 \u0631\u0633\u0648\u0645 \u0627\u0644\u062c\u0648\u0627\u0632", elem_classes=["chip-btn"])
                    chip2 = gr.Button("\U0001f4cb \u062a\u062c\u062f\u064a\u062f \u0627\u0644\u0647\u0648\u064a\u0629", elem_classes=["chip-btn"])
                    chip3 = gr.Button("\U0001f697 \u0627\u0644\u0645\u062e\u0627\u0644\u0641\u0627\u062a \u0627\u0644\u0645\u0631\u0648\u0631\u064a\u0629", elem_classes=["chip-btn"])
                    chip4 = gr.Button("\U0001f3e0 \u062e\u0631\u0648\u062c \u0648\u0639\u0648\u062f\u0629", elem_classes=["chip-btn"])

                # Input Bar
                with gr.Row(elem_classes=["modern-input-row"]):
                    clear_btn = gr.Button("\U0001f5d1\ufe0f", elem_classes=["icon-btn"])
                    tts_btn = gr.Button("\U0001f50a", elem_classes=["icon-btn"])
                    msg_input = gr.Textbox(
                        placeholder="\u0627\u0643\u062a\u0628 \u0627\u0633\u062a\u0641\u0633\u0627\u0631\u0643 \u0647\u0646\u0627...",
                        show_label=False,
                        lines=1,
                        elem_id="msg-input",
                        scale=10
                    )
                    submit_btn = gr.Button("\u27a4", variant="primary", elem_classes=["icon-btn", "send-btn"])

                with gr.Accordion("\U0001f3a4 Voice Input / \u0627\u0644\u0625\u062f\u062e\u0627\u0644 \u0627\u0644\u0635\u0648\u062a\u064a", open=False, elem_classes=["voice-accordion"]):
                    audio_input = gr.Audio(source="microphone", type="filepath", label=None)

                tts_player = gr.Audio(autoplay=True, visible=False)

            # Sidebar
            with gr.Column(elem_classes=["controls-panel"]):
                gr.Markdown("## \u2699\ufe0f \u0627\u0644\u0625\u0639\u062f\u0627\u062f\u0627\u062a")
                lang_dropdown = gr.Dropdown(
                    choices=SUPPORTED_LANGUAGES,
                    label="Language / \u0627\u0644\u0644\u063a\u0629"
                )
                gr.HTML("<div style='height:1px;background:var(--border);margin:16px 0;'></div>")
                gr.Markdown("### \u0645\u0639\u0644\u0648\u0645\u0627\u062a \u0627\u0644\u062c\u0644\u0633\u0629")
                gr.HTML(f"""
                    <div class='info-card'>
                        <p><strong>Hardware:</strong> NVIDIA A100-80GB</p>
                        <p><strong>Precision:</strong> {Config.TORCH_DTYPE}</p>
                        <p><strong>Model:</strong> ALLaM-7B</p>
                        <p><strong>Retrieval:</strong> Hybrid (FAISS + BM25)</p>
                    </div>
                """)
                gr.HTML("<div style='height:1px;background:var(--border);margin:16px 0;'></div>")
                close_sidebar = gr.Button("\u0625\u063a\u0644\u0627\u0642 | Close", variant="secondary", elem_classes=["close-btn"])

        # --- 4. EVENTS ---

        enter_btn.click(
            fn=enter_system,
            inputs=[init_lang_radio],
            outputs=[welcome_container, main_container, lang_dropdown]
        ).then(fn=None, inputs=[init_lang_radio], outputs=None, _js=SET_DIRECTION_JS)

        # Language change in sidebar also switches direction
        lang_dropdown.change(fn=None, inputs=[lang_dropdown], outputs=None, _js=SET_DIRECTION_JS)

        settings_btn.click(None, None, None, _js=SIDEBAR_TOGGLE_JS)
        close_sidebar.click(None, None, None, _js=SIDEBAR_TOGGLE_JS)
        theme_btn.click(None, None, None, _js=TOGGLE_JS)

        submit_event = {
            "fn": chat_pipeline,
            "inputs": [msg_input, chatbot, audio_input, lang_dropdown],
            "outputs": [chatbot, tts_player, msg_input, audio_input]
        }
        submit_btn.click(**submit_event)
        msg_input.submit(**submit_event)

        for chip, text in [
            (chip1, "\u0643\u0645 \u0631\u0633\u0648\u0645 \u0625\u0635\u062f\u0627\u0631 \u062c\u0648\u0627\u0632 \u0627\u0644\u0633\u0641\u0631\u061f"),
            (chip2, "\u0643\u064a\u0641 \u0623\u062c\u062f\u062f \u0628\u0637\u0627\u0642\u0629 \u0627\u0644\u0647\u0648\u064a\u0629 \u0627\u0644\u0648\u0637\u0646\u064a\u0629\u061f"),
            (chip3, "\u0643\u064a\u0641 \u0623\u0633\u062a\u0639\u0644\u0645 \u0639\u0646 \u0627\u0644\u0645\u062e\u0627\u0644\u0641\u0627\u062a \u0627\u0644\u0645\u0631\u0648\u0631\u064a\u0629\u061f"),
            (chip4, "\u0645\u0627 \u0647\u064a \u062e\u0637\u0648\u0627\u062a \u0625\u0635\u062f\u0627\u0631 \u062a\u0623\u0634\u064a\u0631\u0629 \u062e\u0631\u0648\u062c \u0648\u0639\u0648\u062f\u0629\u061f"),
        ]:
            chip.click(fn=lambda t=text: t, inputs=None, outputs=[msg_input])

        tts_btn.click(trigger_tts, [chatbot], [tts_player])
        clear_btn.click(lambda: ([], None, "", None), None, [chatbot, tts_player, msg_input, audio_input])

        demo.load(None, None, None, _js=THEME_JS)

    return demo