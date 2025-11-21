import gradio as gr
from core.model_loader import ModelManager
from ui.theme import MOI_CSS, HEADER_HTML

def create_app(rag_chain):
    """
    Builds and returns the Gradio Blocks app.
    Receives the initialized 'rag_chain' from main.py.
    """
    
    # --- Logic Handlers ---
    def start_chat(lang):
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

    def chat_response(message, history, audio_file, lang_val):
        # 1. Handle Audio if present
        if audio_file:
            asr_pipe = ModelManager.get_asr_pipeline()
            if asr_pipe:
                target_lang = "ar" if lang_val == "Arabic" else "en"
                try:
                    text = asr_pipe(audio_file, generate_kwargs={"language": target_lang})["text"].strip()
                    message = text
                    user_display = f"🎤 {text}"
                except Exception as e:
                    return history + [[None, f"❌ Audio Error: {str(e)}"]]
            else:
                return history + [[None, "⚠️ Error: Whisper model not loaded."]]
        else:
            user_display = message

        if not message: return history

        # 2. Get Answer from RAG Chain
        try:
            response = rag_chain.answer(message)
        except Exception as e:
            response = f"❌ System Error: {str(e)}"

        history.append((user_display, response))
        return history

    def reset_app():
        return (
            gr.update(visible=True),  # Show Welcome
            gr.update(visible=False), # Hide Chat
            None,                     # Clear State
            []                        # Clear History
        )

    def clear_inputs(): return "", None

    # --- UI Layout ---
    with gr.Blocks(theme=gr.themes.Soft(), css=MOI_CSS, title="MOI Assistant") as demo:
        lang_state = gr.State(value="Arabic")

        gr.HTML(HEADER_HTML)

        # SCREEN 1: Welcome
        with gr.Group(visible=True) as welcome_screen:
            gr.Markdown("<h3 style='text-align: center;'>🌍 Please select your preferred language</h3>")
            with gr.Row():
                btn_ar = gr.Button("العربية 🇸🇦", variant="primary", elem_classes=["lang-btn"])
                btn_en = gr.Button("English 🇬🇧", variant="secondary", elem_classes=["lang-btn"])

        # SCREEN 2: Chat
        with gr.Group(visible=False) as chat_screen:
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="Chat", height=500)
                    with gr.Row():
                        msg = gr.Textbox(show_label=False, container=False, scale=4)
                        submit_btn = gr.Button("🚀", variant="primary", scale=1)

                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ أدوات / Tools")
                    audio_input = gr.Audio(source="microphone", type="filepath", label="Voice Input")
                    gr.Markdown("---")
                    restart_btn = gr.Button("🔄 Change Language", variant="secondary")

        # --- Event Wiring ---
        btn_ar.click(fn=lambda: start_chat("Arabic"), outputs=[welcome_screen, chat_screen, chatbot, lang_state, chatbot, msg])
        btn_en.click(fn=lambda: start_chat("English"), outputs=[welcome_screen, chat_screen, chatbot, lang_state, chatbot, msg])

        msg.submit(chat_response, [msg, chatbot, audio_input, lang_state], [chatbot]).then(clear_inputs, None, [msg, audio_input])
        submit_btn.click(chat_response, [msg, chatbot, audio_input, lang_state], [chatbot]).then(clear_inputs, None, [msg, audio_input])
        restart_btn.click(reset_app, outputs=[welcome_screen, chat_screen, lang_state, chatbot])

    return demo