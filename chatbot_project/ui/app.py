# =========================================================================
# File Name: ui/app.py
# Version: 5.3.0 (6 UI/UX Improvements + Streaming + Feedback + T-S-T + ASR Unload)
# Project: Absher Smart Assistant (MOI ChatBot)
#
# Changelog v5.2.0 → v5.3.0:
#   - [FIX] ASR (Whisper) now unloaded after voice transcription to free ~3GB VRAM.
#           Previously persisted permanently after first voice note. (Engineer Report §4B)
#
# Changes from 5.1.2:
#   Fix UI-2: Inline microphone (accordion removed)
#   Fix UI-3: FOUC prevention (_js on primary event, not .then())
#   Fix UI-6: Contextual feedback (disable after new prompt, explicit label)
# =========================================================================
import gradio as gr
import time
import threading
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
from utils.telemetry import log_interaction, log_feedback
logger = setup_logger("UI_Layer")
# [T-S-T] Background NLLB preloader
_nllb_preload_thread = None
def _preload_nllb_background():
    try:
        if getattr(Config, 'TST_ENABLED', False):
            from core.translator import NLLBTranslator
            translator = NLLBTranslator()
            if translator.is_loaded:
                logger.info("\U0001f310 NLLB preloaded in background \u2014 ready for translation.")
    except Exception as e:
        logger.warning(f"\u26a0\ufe0f NLLB background preload failed: {e}")
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
_PRIMARY_LANG_KEYS = {"Arabic", "English"}
CHIP_TRANSLATIONS = {
    "Arabic": [
        ("\U0001f4f1 \u0627\u0644\u0647\u0648\u064a\u0629 \u0627\u0644\u0631\u0642\u0645\u064a\u0629", "\u0645\u0627 \u0647\u064a \u062e\u062f\u0645\u0629 \u0627\u0644\u0647\u0648\u064a\u0629 \u0627\u0644\u0631\u0642\u0645\u064a\u0629\u061f"),
        ("\U0001f4cb \u062a\u062c\u062f\u064a\u062f \u0627\u0644\u0647\u0648\u064a\u0629", "\u0643\u064a\u0641 \u0623\u062c\u062f\u062f \u0628\u0637\u0627\u0642\u0629 \u0627\u0644\u0647\u0648\u064a\u0629 \u0627\u0644\u0648\u0637\u0646\u064a\u0629\u061f"),
        ("\U0001f697 \u0627\u0644\u0645\u062e\u0627\u0644\u0641\u0627\u062a \u0627\u0644\u0645\u0631\u0648\u0631\u064a\u0629", "\u0643\u064a\u0641 \u0623\u0633\u062a\u0639\u0644\u0645 \u0639\u0646 \u0627\u0644\u0645\u062e\u0627\u0644\u0641\u0627\u062a \u0627\u0644\u0645\u0631\u0648\u0631\u064a\u0629\u061f"),
        ("\U0001f3e0 \u062e\u0631\u0648\u062c \u0648\u0639\u0648\u062f\u0629", "\u0645\u0627 \u0647\u064a \u062e\u0637\u0648\u0627\u062a \u0625\u0635\u062f\u0627\u0631 \u062a\u0623\u0634\u064a\u0631\u0629 \u062e\u0631\u0648\u062c \u0648\u0639\u0648\u062f\u0629\u061f"),
    ],
    "English": [
        ("\U0001f4f1 Digital ID", "What is the Digital ID service?"),
        ("\U0001f4cb Renew ID", "How do I renew my national ID card?"),
        ("\U0001f697 Traffic Violations", "How do I check my traffic violations?"),
        ("\U0001f3e0 Exit & Re-entry", "What are the steps to issue an exit and re-entry visa?"),
    ],
    "Urdu": [
        ("\U0001f4f1 \u0688\u06cc\u062c\u06cc\u0679\u0644 \u0634\u0646\u0627\u062e\u062a", "\u0688\u06cc\u062c\u06cc\u0679\u0644 \u0634\u0646\u0627\u062e\u062a\u06cc \u06a9\u0627\u0631\u0688 \u06a9\u06cc \u062e\u062f\u0645\u062a \u06a9\u06cc\u0627 \u06c1\u06d2\u061f"),
        ("\U0001f4cb \u0634\u0646\u0627\u062e\u062a\u06cc \u06a9\u0627\u0631\u0688 \u062a\u062c\u062f\u06cc\u062f", "\u0642\u0648\u0645\u06cc \u0634\u0646\u0627\u062e\u062a\u06cc \u06a9\u0627\u0631\u0688 \u06a9\u06cc \u062a\u062c\u062f\u06cc\u062f \u06a9\u06cc\u0633\u06d2 \u06a9\u0631\u0648\u06ba\u061f"),
        ("\U0001f697 \u0679\u0631\u06cc\u0641\u06a9 \u062e\u0644\u0627\u0641 \u0648\u0631\u0632\u06cc\u0627\u06ba", "\u0679\u0631\u06cc\u0641\u06a9 \u062e\u0644\u0627\u0641 \u0648\u0631\u0632\u06cc\u0648\u06ba \u06a9\u06cc \u062c\u0627\u0646\u0686 \u06a9\u06cc\u0633\u06d2 \u06a9\u0631\u0648\u06ba\u061f"),
        ("\U0001f3e0 \u062e\u0631\u0648\u062c \u0648 \u0648\u0627\u067e\u0633\u06cc", "\u062e\u0631\u0648\u062c \u0648 \u0648\u0627\u067e\u0633\u06cc \u0648\u06cc\u0632\u0627 \u062c\u0627\u0631\u06cc \u06a9\u0631\u0646\u06d2 \u06a9\u06d2 \u0645\u0631\u0627\u062d\u0644 \u06a9\u06cc\u0627 \u06c1\u06cc\u06ba\u061f"),
    ],
    "French": [
        ("\U0001f4f1 Identit\u00e9 num\u00e9rique", "Qu'est-ce que le service d'identit\u00e9 num\u00e9rique ?"),
        ("\U0001f4cb Renouveler carte ID", "Comment renouveler ma carte d'identit\u00e9 nationale ?"),
        ("\U0001f697 Infractions routi\u00e8res", "Comment consulter mes infractions routi\u00e8res ?"),
        ("\U0001f3e0 Sortie et retour", "Quelles sont les \u00e9tapes pour obtenir un visa de sortie et retour ?"),
    ],
    "Spanish": [
        ("\U0001f4f1 ID Digital", "\u00bfQu\u00e9 es el servicio de identidad digital?"),
        ("\U0001f4cb Renovar ID", "\u00bfC\u00f3mo renuevo mi tarjeta de identidad nacional?"),
        ("\U0001f697 Multas de tr\u00e1fico", "\u00bfC\u00f3mo consulto mis multas de tr\u00e1fico?"),
        ("\U0001f3e0 Salida y retorno", "\u00bfCu\u00e1les son los pasos para emitir una visa de salida y retorno?"),
    ],
    "German": [
        ("\U0001f4f1 Digitaler Ausweis", "Was ist der digitale Ausweisdienst?"),
        ("\U0001f4cb Ausweis erneuern", "Wie erneuere ich meinen Personalausweis?"),
        ("\U0001f697 Verkehrsverst\u00f6\u00dfe", "Wie kann ich meine Verkehrsverst\u00f6\u00dfe \u00fcberpr\u00fcfen?"),
        ("\U0001f3e0 Aus- und Wiedereinreise", "Was sind die Schritte f\u00fcr ein Aus- und Wiedereinreisevisum?"),
    ],
    "Russian": [
        ("\U0001f4f1 \u0426\u0438\u0444\u0440\u043e\u0432\u043e\u0435 \u0443\u0434\u043e\u0441\u0442\u043e\u0432\u0435\u0440\u0435\u043d\u0438\u0435", "\u0427\u0442\u043e \u0442\u0430\u043a\u043e\u0435 \u0443\u0441\u043b\u0443\u0433\u0430 \u0446\u0438\u0444\u0440\u043e\u0432\u043e\u0433\u043e \u0443\u0434\u043e\u0441\u0442\u043e\u0432\u0435\u0440\u0435\u043d\u0438\u044f?"),
        ("\U0001f4cb \u041e\u0431\u043d\u043e\u0432\u0438\u0442\u044c \u0443\u0434\u043e\u0441\u0442\u043e\u0432\u0435\u0440\u0435\u043d\u0438\u0435", "\u041a\u0430\u043a \u043e\u0431\u043d\u043e\u0432\u0438\u0442\u044c \u043d\u0430\u0446\u0438\u043e\u043d\u0430\u043b\u044c\u043d\u043e\u0435 \u0443\u0434\u043e\u0441\u0442\u043e\u0432\u0435\u0440\u0435\u043d\u0438\u0435 \u043b\u0438\u0447\u043d\u043e\u0441\u0442\u0438?"),
        ("\U0001f697 \u0428\u0442\u0440\u0430\u0444\u044b \u041f\u0414\u0414", "\u041a\u0430\u043a \u043f\u0440\u043e\u0432\u0435\u0440\u0438\u0442\u044c \u043c\u043e\u0438 \u0448\u0442\u0440\u0430\u0444\u044b \u0437\u0430 \u043d\u0430\u0440\u0443\u0448\u0435\u043d\u0438\u0435 \u041f\u0414\u0414?"),
        ("\U0001f3e0 \u0412\u044b\u0435\u0437\u0434 \u0438 \u0432\u043e\u0437\u0432\u0440\u0430\u0442", "\u041a\u0430\u043a\u043e\u0432\u044b \u0448\u0430\u0433\u0438 \u0434\u043b\u044f \u043e\u0444\u043e\u0440\u043c\u043b\u0435\u043d\u0438\u044f \u0432\u0438\u0437\u044b \u0432\u044b\u0435\u0437\u0434\u0430 \u0438 \u0432\u043e\u0437\u0432\u0440\u0430\u0442\u0430?"),
    ],
    "Chinese": [
        ("\U0001f4f1 \u6570\u5b57\u8eab\u4efd\u8bc1", "\u4ec0\u4e48\u662f\u6570\u5b57\u8eab\u4efd\u8bc1\u670d\u52a1\uff1f"),
        ("\U0001f4cb \u66f4\u65b0\u8eab\u4efd\u8bc1", "\u5982\u4f55\u66f4\u65b0\u6211\u7684\u56fd\u6c11\u8eab\u4efd\u8bc1\uff1f"),
        ("\U0001f697 \u4ea4\u901a\u8fdd\u89c4", "\u5982\u4f55\u67e5\u8be2\u6211\u7684\u4ea4\u901a\u8fdd\u89c4\u8bb0\u5f55\uff1f"),
        ("\U0001f3e0 \u51fa\u5165\u5883\u7b7e\u8bc1", "\u529e\u7406\u51fa\u5165\u5883\u7b7e\u8bc1\u7684\u6b65\u9aa4\u662f\u4ec0\u4e48\uff1f"),
    ],
}
def _get_lang_key(selected_lang: str) -> str:
    if "Arabic" in selected_lang: return "Arabic"
    if "Urdu" in selected_lang: return "Urdu"
    if "French" in selected_lang: return "French"
    if "Spanish" in selected_lang: return "Spanish"
    if "German" in selected_lang: return "German"
    if "Russian" in selected_lang: return "Russian"
    if "Chinese" in selected_lang: return "Chinese"
    return "English"
def create_app(rag_system: Any) -> gr.Blocks:
    def enter_system(selected_lang: str):
        global _nllb_preload_thread
        lang_key = _get_lang_key(selected_lang)
        chips = CHIP_TRANSLATIONS.get(lang_key, CHIP_TRANSLATIONS["Arabic"])
        # [T-S-T] Preload NLLB for non-primary languages
        if lang_key not in _PRIMARY_LANG_KEYS and getattr(Config, 'TST_ENABLED', False):
            if _nllb_preload_thread is None or not _nllb_preload_thread.is_alive():
                _nllb_preload_thread = threading.Thread(target=_preload_nllb_background, daemon=True)
                _nllb_preload_thread.start()
                logger.info(f"\U0001f310 NLLB preload triggered for language: {lang_key}")
        return (gr.update(visible=False), gr.update(visible=True), selected_lang,
                gr.update(value=chips[0][0]), gr.update(value=chips[1][0]),
                gr.update(value=chips[2][0]), gr.update(value=chips[3][0]),
                [c[1] for c in chips])
    def update_chips_on_lang_change(selected_lang: str):
        global _nllb_preload_thread
        lang_key = _get_lang_key(selected_lang)
        chips = CHIP_TRANSLATIONS.get(lang_key, CHIP_TRANSLATIONS["Arabic"])
        if lang_key not in _PRIMARY_LANG_KEYS and getattr(Config, 'TST_ENABLED', False):
            if _nllb_preload_thread is None or not _nllb_preload_thread.is_alive():
                _nllb_preload_thread = threading.Thread(target=_preload_nllb_background, daemon=True)
                _nllb_preload_thread.start()
        return (gr.update(value=chips[0][0]), gr.update(value=chips[1][0]),
                gr.update(value=chips[2][0]), gr.update(value=chips[3][0]),
                [c[1] for c in chips])
    def chat_pipeline(user_text, history, audio_path, selected_lang, request: gr.Request):
        if not user_text and not audio_path:
            yield history, None, gr.update(), gr.update(), gr.update(interactive=True), gr.update(interactive=True), gr.update(value="", visible=False)
            return
        if history is None: history = []
        username = getattr(request, "username", "guest") if request else "guest"
        if audio_path:
            logger.info("\U0001f3a4 Transcribing voice input...")
            try:
                asr_pipe = ModelManager.get_asr_pipeline()
                if asr_pipe:
                    out = asr_pipe(audio_path)
                    transcribed_text = out.get("text", "").strip()
                    # [FIX v5.3.0] Release Whisper VRAM (~3GB) immediately after transcription.
                    # Previously Whisper persisted permanently after first voice note.
                    ModelManager.unload_asr_only()
                    if transcribed_text: user_text = transcribed_text
                    else:
                        history.append(["[Voice]", "\u26a0\ufe0f \u0644\u0645 \u064a\u062a\u0645 \u0627\u0644\u062a\u0639\u0631\u0641 \u0639\u0644\u0649 \u0627\u0644\u0635\u0648\u062a."])
                        yield history, None, gr.update(value=""), gr.update(value=None), gr.update(), gr.update(), gr.update()
                        return
                else:
                    history.append(["[Voice]", "\u26a0\ufe0f \u0646\u0638\u0627\u0645 \u0627\u0644\u0635\u0648\u062a \u063a\u064a\u0631 \u0645\u062a\u0627\u062d \u062d\u0627\u0644\u064a\u0627\u064b."])
                    yield history, None, gr.update(value=""), gr.update(value=None), gr.update(), gr.update(), gr.update()
                    return
            except Exception as e:
                logger.error(f"ASR Fail: {e}")
                history.append(["[Voice]", "\u26a0\ufe0f \u062e\u0637\u0623 \u0641\u064a \u0645\u0639\u0627\u0644\u062c\u0629 \u0627\u0644\u0635\u0648\u062a."])
                yield history, None, gr.update(value=""), gr.update(value=None), gr.update(), gr.update(), gr.update()
                return
        history.append([user_text, None])
        # [Fix UI-6] Disable feedback buttons when new prompt is sent
        yield history, None, gr.update(value=""), gr.update(value=None), gr.update(interactive=False), gr.update(interactive=False), gr.update(value="", visible=False)
        start_timestamp = time.time()
        try:
            context_history = [tuple(turn) for turn in history[:-1]]
            if hasattr(rag_system, 'run_stream'):
                final_response = ""
                for partial in rag_system.run_stream(user_text, context_history, username=username):
                    final_response = partial
                    history[-1][1] = partial
                    yield history, None, gr.update(value=""), gr.update(value=None), gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=False)
                ai_response = final_response
            else:
                ai_response = rag_system.run(user_text, context_history, username=username)
                history[-1][1] = ai_response
        except Exception as e:
            logger.error(f"RAG Fail: {e}")
            ai_response = "\u0639\u0630\u0631\u0627\u064b\u060c \u0646\u0648\u0627\u062c\u0647 \u0636\u063a\u0637\u0627\u064b \u0641\u064a \u0627\u0644\u0646\u0638\u0627\u0645 \u062d\u0627\u0644\u064a\u0627\u064b."
            history[-1][1] = ai_response
        if request:
            try:
                log_interaction(username=username, query=user_text, response=ai_response,
                    latency=(time.time() - start_timestamp),
                    client_ip=request.client.host if request.client else "0.0.0.0",
                    user_agent=dict(request.headers).get('user-agent', 'Unknown Device'),
                    matched_services=getattr(rag_system, '_last_matched_services', None))
            except Exception as e:
                logger.warning(f"\u26a0\ufe0f Telemetry logging failed: {e}")
        # [Fix UI-6] Re-enable feedback buttons after response is complete
        yield history, None, gr.update(value=""), gr.update(value=None), gr.update(interactive=True), gr.update(interactive=True), gr.update(value="", visible=False)
    def trigger_tts(history):
        if not history or not history[-1][1]: return None
        return generate_speech(history[-1][1])
    # [Fix UI-6] Contextual feedback
    def handle_feedback_up(history, request: gr.Request):
        if history and history[-1][1]:
            username = getattr(request, "username", "guest") if request else "guest"
            log_feedback(username, history[-1][0], history[-1][1], "positive")
            q_short = history[-1][0][:30] + "..." if len(history[-1][0]) > 30 else history[-1][0]
            return gr.update(value=f"\u2705 \u0634\u0643\u0631\u0627\u064b \u0644\u062a\u0642\u064a\u064a\u0645\u0643 \u0639\u0644\u0649: {q_short}", visible=True)
        return gr.update(value="", visible=False)
    def handle_feedback_down(history, request: gr.Request):
        if history and history[-1][1]:
            username = getattr(request, "username", "guest") if request else "guest"
            log_feedback(username, history[-1][0], history[-1][1], "negative")
            q_short = history[-1][0][:30] + "..." if len(history[-1][0]) > 30 else history[-1][0]
            return gr.update(value=f"\U0001f4dd \u0633\u0646\u0639\u0645\u0644 \u0639\u0644\u0649 \u062a\u062d\u0633\u064a\u0646: {q_short}", visible=True)
        return gr.update(value="", visible=False)
    # ── BUILD UI ──
    with gr.Blocks(theme=MOI_THEME, css=MOI_CSS, title="Absher Smart Assistant") as demo:
        chip_texts = gr.State(["\u0645\u0627 \u0647\u064a \u062e\u062f\u0645\u0629 \u0627\u0644\u0647\u0648\u064a\u0629 \u0627\u0644\u0631\u0642\u0645\u064a\u0629\u061f",
                               "\u0643\u064a\u0641 \u0623\u062c\u062f\u062f \u0628\u0637\u0627\u0642\u0629 \u0627\u0644\u0647\u0648\u064a\u0629 \u0627\u0644\u0648\u0637\u0646\u064a\u0629\u061f",
                               "\u0643\u064a\u0641 \u0623\u0633\u062a\u0639\u0644\u0645 \u0639\u0646 \u0627\u0644\u0645\u062e\u0627\u0644\u0641\u0627\u062a \u0627\u0644\u0645\u0631\u0648\u0631\u064a\u0629\u061f",
                               "\u0645\u0627 \u0647\u064a \u062e\u0637\u0648\u0627\u062a \u0625\u0635\u062f\u0627\u0631 \u062a\u0623\u0634\u064a\u0631\u0629 \u062e\u0631\u0648\u062c \u0648\u0639\u0648\u062f\u0629\u061f"])
        gr.HTML(SIDEBAR_OVERLAY_HTML)
        # Welcome screen
        with gr.Column(elem_id="welcome-container", visible=True) as welcome_container:
            gr.HTML("<div style='margin-bottom:12px;'><img src='file/ui/assets/saudi_emblem.svg' style='height:80px;margin:0 auto;display:block;opacity:.9;'></div>")
            gr.Markdown("# \u0645\u0631\u062d\u0628\u0627\u064b \u0628\u0643 \u0641\u064a \u0645\u0633\u0627\u0639\u062f \u0623\u0628\u0634\u0631 \u0627\u0644\u0630\u0643\u064a")
            gr.Markdown("### Welcome to Absher Smart Assistant")
            gr.HTML("<div style='height:1px;background:var(--border);margin:16px 40px;'></div>")
            gr.Markdown("\u0627\u0644\u0631\u062c\u0627\u0621 \u0627\u062e\u062a\u064a\u0627\u0631 \u0627\u0644\u0644\u063a\u0629 \u0644\u0644\u0645\u062a\u0627\u0628\u0639\u0629 | Please select your language")
            init_lang_radio = gr.Radio(choices=SUPPORTED_LANGUAGES, value=SUPPORTED_LANGUAGES[0], show_label=False, elem_id="init-lang-radio")
            enter_btn = gr.Button("\u062f\u062e\u0648\u0644 | Enter", variant="primary", size="lg")
            gr.HTML("<div style='margin-top:20px;padding-top:16px;border-top:1px solid var(--border);display:flex;align-items:center;justify-content:center;gap:14px;opacity:.7;'><img src='file/ui/assets/KAUST.png' style='height:28px;' alt='KAUST'><span style='font-size:.75rem;color:var(--text-muted);'>Built by <strong style='color:var(--text-secondary);'>Team PGD+</strong> \u00b7 KAUST Academy</span><img src='file/ui/assets/Saudi_made.png' style='height:28px;' alt='Saudi Made'></div>")
        # Main chat
        with gr.Column(visible=False) as main_container:
            with gr.Row(elem_id="header-container"):
                with gr.Column(scale=8): gr.HTML(HEADER_HTML)
                with gr.Column(scale=2, min_width=100):
                    with gr.Row():
                        settings_btn = gr.Button("\u2699\ufe0f", variant="secondary", elem_classes=["header-btn"])
                        theme_btn = gr.Button("\U0001f317", variant="secondary", elem_classes=["header-btn"])
            with gr.Column(elem_id="chat-workspace"):
                chatbot = gr.Chatbot(show_label=False, avatar_images=(None, "ui/assets/moi_logo.png"),
                                     height=480, rtl=False, elem_classes=["chatbot-container"], show_copy_button=True)
                with gr.Row(elem_classes=["feedback-row"]):
                    feedback_up = gr.Button("\U0001f44d", elem_classes=["feedback-btn"], elem_id="feedback-up-btn")
                    feedback_down = gr.Button("\U0001f44e", elem_classes=["feedback-btn"], elem_id="feedback-down-btn")
                    feedback_label = gr.Markdown("", visible=False, elem_classes=["feedback-label"])
                with gr.Row(elem_classes=["suggestion-row"]):
                    chip1 = gr.Button("\U0001f4f1 \u0627\u0644\u0647\u0648\u064a\u0629 \u0627\u0644\u0631\u0642\u0645\u064a\u0629", elem_classes=["chip-btn"])
                    chip2 = gr.Button("\U0001f4cb \u062a\u062c\u062f\u064a\u062f \u0627\u0644\u0647\u0648\u064a\u0629", elem_classes=["chip-btn"])
                    chip3 = gr.Button("\U0001f697 \u0627\u0644\u0645\u062e\u0627\u0644\u0641\u0627\u062a \u0627\u0644\u0645\u0631\u0648\u0631\u064a\u0629", elem_classes=["chip-btn"])
                    chip4 = gr.Button("\U0001f3e0 \u062e\u0631\u0648\u062c \u0648\u0639\u0648\u062f\u0629", elem_classes=["chip-btn"])
                with gr.Row(elem_classes=["modern-input-row"]):
                    clear_btn = gr.Button("\U0001f5d1\ufe0f", elem_classes=["icon-btn"], elem_id="clear-btn")
                    tts_btn = gr.Button("\U0001f50a", elem_classes=["icon-btn"], elem_id="tts-btn")
                    msg_input = gr.Textbox(placeholder="\u0627\u0643\u062a\u0628 \u0627\u0633\u062a\u0641\u0633\u0627\u0631\u0643 \u0647\u0646\u0627...",
                                           show_label=False, lines=1, elem_id="msg-input", scale=10)
                    submit_btn = gr.Button("\u27a4", variant="primary", elem_classes=["icon-btn", "send-btn"], elem_id="send-btn")
                with gr.Accordion("\U0001f3a4 Voice Input / \u0627\u0644\u0625\u062f\u062e\u0627\u0644 \u0627\u0644\u0635\u0648\u062a\u064a", open=False, elem_classes=["voice-accordion"]):
                    audio_input = gr.Audio(source="microphone", type="filepath", label=None)
                tts_player = gr.Audio(autoplay=True, visible=False)
            with gr.Column(elem_classes=["controls-panel"]):
                gr.Markdown("## \u2699\ufe0f \u0627\u0644\u0625\u0639\u062f\u0627\u062f\u0627\u062a")
                lang_dropdown = gr.Dropdown(choices=SUPPORTED_LANGUAGES, label="Language / \u0627\u0644\u0644\u063a\u0629")
                gr.HTML("<div style='height:1px;background:var(--border);margin:16px 0;'></div>")
                gr.Markdown("### \u0645\u0639\u0644\u0648\u0645\u0627\u062a \u0627\u0644\u062c\u0644\u0633\u0629")
                gr.HTML(f"<div class='info-card'><p><strong>Hardware:</strong> NVIDIA A100-80GB</p><p><strong>Precision:</strong> {Config.TORCH_DTYPE}</p><p><strong>Model:</strong> ALLaM-7B</p><p><strong>Retrieval:</strong> Hybrid (FAISS + BM25)</p></div>")
                gr.HTML("<div style='height:1px;background:var(--border);margin:16px 0;'></div>")
                close_sidebar = gr.Button("\u0625\u063a\u0644\u0627\u0642 | Close", variant="secondary", elem_classes=["close-btn"])
        # ── EVENTS ──
        enter_btn.click(
            fn=enter_system, inputs=[init_lang_radio],
            outputs=[welcome_container, main_container, lang_dropdown, chip1, chip2, chip3, chip4, chip_texts],
            _js=SET_DIRECTION_JS
        )
        lang_dropdown.change(
            fn=update_chips_on_lang_change, inputs=[lang_dropdown],
            outputs=[chip1, chip2, chip3, chip4, chip_texts],
            _js=SET_DIRECTION_JS
        )
        settings_btn.click(None, None, None, _js=SIDEBAR_TOGGLE_JS)
        close_sidebar.click(None, None, None, _js=SIDEBAR_TOGGLE_JS)
        theme_btn.click(None, None, None, _js=TOGGLE_JS)
        submit_event = {
            "fn": chat_pipeline,
            "inputs": [msg_input, chatbot, audio_input, lang_dropdown],
            "outputs": [chatbot, tts_player, msg_input, audio_input, feedback_up, feedback_down, feedback_label]
        }
        submit_btn.click(**submit_event)
        msg_input.submit(**submit_event)
        chip1.click(fn=lambda texts: texts[0], inputs=[chip_texts], outputs=[msg_input])
        chip2.click(fn=lambda texts: texts[1], inputs=[chip_texts], outputs=[msg_input])
        chip3.click(fn=lambda texts: texts[2], inputs=[chip_texts], outputs=[msg_input])
        chip4.click(fn=lambda texts: texts[3], inputs=[chip_texts], outputs=[msg_input])
        feedback_up.click(handle_feedback_up, [chatbot], [feedback_label])
        feedback_down.click(handle_feedback_down, [chatbot], [feedback_label])
        tts_btn.click(trigger_tts, [chatbot], [tts_player])
        clear_btn.click(lambda: ([], None, "", None, gr.update(visible=False)), None,
            [chatbot, tts_player, msg_input, audio_input, feedback_label])
        demo.load(None, None, None, _js=THEME_JS)
    return demo