# =========================================================================
# File Name: ui/theme.py
# Version: 5.2.0 (6 UI/UX Improvements)
# Project: Absher Smart Assistant (MOI ChatBot)
# Changes from 5.1.2:
#   Fix UI-1: Markdown/table/list alignment in chat bubbles (CSS)
#   Fix UI-2: Inline microphone button CSS (accordion removed in app.py)
#   Fix UI-3: FOUC prevention — instant dir switch via _js on primary event
#   Fix UI-4: Smooth auto-scroll during LLM streaming (MutationObserver)
#   Fix UI-5: Accessibility (aria-labels for icon buttons)
#   Fix UI-6: Contextual feedback styling
# =========================================================================
import gradio as gr

# =========================================================================
# 1. JAVASCRIPT
# =========================================================================

THEME_JS = """
function() {
    var savedTheme = localStorage.getItem('moi_theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark');
    } else if (!savedTheme) {
        var hour = new Date().getHours();
        if (hour < 6 || hour >= 18) document.body.classList.add('dark');
    }
    document.documentElement.setAttribute('dir', 'rtl');
    document.body.classList.add('is-rtl');
    document.body.classList.remove('is-ltr');
    window.toggleMoiSidebar = function() {
        var sidebar = document.querySelector('.controls-panel');
        var overlay = document.querySelector('.sidebar-overlay');
        if (sidebar) sidebar.classList.toggle('sidebar-open');
        if (overlay) overlay.classList.toggle('overlay-active');
    };
    window.applyDirection = function(dir) {
        var isRTL = (dir === 'rtl');
        document.documentElement.setAttribute('dir', dir);
        document.body.setAttribute('dir', dir);
        document.body.classList.toggle('is-rtl', isRTL);
        document.body.classList.toggle('is-ltr', !isRTL);
    };
    window.applyDirection('rtl');
    var resizeTimer;
    function setVH() {
        document.documentElement.style.setProperty('--vh', window.innerHeight * 0.01 + 'px');
    }
    setVH();
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(setVH, 100);
    });
    /* [Fix UI-4] Auto-scroll + direction observer */
    var observer = new MutationObserver(function(mutations) {
        var dir = document.documentElement.getAttribute('dir') || 'rtl';
        var didAdd = false;
        for (var i = 0; i < mutations.length; i++) {
            var added = mutations[i].addedNodes;
            for (var j = 0; j < added.length; j++) {
                var node = added[j];
                if (node.nodeType === 1 && node.querySelector) {
                    var msgs = node.classList && node.classList.contains('message') ? [node] : node.querySelectorAll('.message');
                    for (var k = 0; k < msgs.length; k++) {
                        msgs[k].setAttribute('dir', dir);
                    }
                    if (msgs.length > 0) didAdd = true;
                }
            }
            if (mutations[i].type === 'characterData' || mutations[i].type === 'childList') {
                didAdd = true;
            }
        }
        if (didAdd) {
            var chatWrap = document.querySelector('.chatbot-container .wrap');
            if (chatWrap) {
                requestAnimationFrame(function() {
                    chatWrap.scrollTo({ top: chatWrap.scrollHeight, behavior: 'smooth' });
                });
            }
        }
    });
    setTimeout(function() {
        var chatWrap = document.querySelector('.chatbot-container .wrap');
        if (chatWrap) observer.observe(chatWrap, { childList: true, subtree: true, characterData: true });
    }, 2000);
    /* [Fix UI-5] Accessibility: inject aria-labels */
    setTimeout(function() {
        var a11yMap = {
            'clear-btn': { 'aria-label': 'Clear chat', 'title': 'Clear chat / \u0645\u0633\u062d \u0627\u0644\u0645\u062d\u0627\u062f\u062b\u0629' },
            'tts-btn': { 'aria-label': 'Listen to response', 'title': 'Text-to-Speech / \u0627\u0633\u062a\u0645\u0639' },
            'send-btn': { 'aria-label': 'Send message', 'title': 'Send / \u0625\u0631\u0633\u0627\u0644' },
            'feedback-up-btn': { 'aria-label': 'Good response', 'title': 'Good response / \u0625\u062c\u0627\u0628\u0629 \u062c\u064a\u062f\u0629' },
            'feedback-down-btn': { 'aria-label': 'Bad response', 'title': 'Bad response / \u0625\u062c\u0627\u0628\u0629 \u0633\u064a\u0626\u0629' }
        };
        for (var id in a11yMap) {
            var el = document.getElementById(id);
            if (el) {
                for (var attr in a11yMap[id]) {
                    el.setAttribute(attr, a11yMap[id][attr]);
                }
            }
        }
        var headerBtns = document.querySelectorAll('.header-btn');
        if (headerBtns.length >= 1) headerBtns[0].setAttribute('aria-label', 'Settings');
        if (headerBtns.length >= 2) headerBtns[1].setAttribute('aria-label', 'Toggle theme');
    }, 3000);
    return [];
}
"""

SET_DIRECTION_JS = """
function(lang) {
    var isRTL = lang.indexOf('\u0627\u0644\u0639\u0631\u0628\u064a\u0629') !== -1 || lang.indexOf('Urdu') !== -1;
    var dir = isRTL ? 'rtl' : 'ltr';
    if (window.applyDirection) {
        window.applyDirection(dir);
    }
    var msgInput = document.querySelector('#msg-input textarea');
    if (msgInput) {
        if (lang.indexOf('\u0627\u0644\u0639\u0631\u0628\u064a\u0629') !== -1) {
            msgInput.setAttribute('placeholder', '\u0627\u0643\u062a\u0628 \u0627\u0633\u062a\u0641\u0633\u0627\u0631\u0643 \u0647\u0646\u0627...');
        } else if (lang.indexOf('Urdu') !== -1) {
            msgInput.setAttribute('placeholder', '\u0627\u067e\u0646\u0627 \u0633\u0648\u0627\u0644 \u06cc\u06c1\u0627\u06ba \u0644\u06a9\u06be\u06cc\u06ba...');
        } else {
            msgInput.setAttribute('placeholder', 'Type your question here...');
        }
    }
    var welcome = document.getElementById('welcome-container');
    if (welcome) {
        welcome.style.opacity = '0';
        welcome.style.transform = 'scale(0.95)';
    }
    return [lang];
}
"""

TOGGLE_JS = """
function() {
    document.body.classList.toggle('dark');
    var isDark = document.body.classList.contains('dark');
    localStorage.setItem('moi_theme', isDark ? 'dark' : 'light');
}
"""

SIDEBAR_TOGGLE_JS = "function() { window.toggleMoiSidebar(); }"


# =========================================================================
# 2. CSS
# =========================================================================
MOI_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;800&display=swap');

:root {
    --primary: #0D3B1E; --primary-light: #1A6B3C;
    --primary-glow: rgba(26,107,60,0.15);
    --gold: #C8A951; --gold-soft: rgba(200,169,81,0.12);
    --bg-page: #F2F4F7; --bg-card: #FFFFFF; --bg-chat: #F7F8FA; --bg-input: #FFFFFF;
    --bubble-user: #0D3B1E; --bubble-user-text: #FFF;
    --bubble-bot: #FFFFFF; --bubble-bot-text: #1A1A2E;
    --text-primary: #1A1A2E; --text-secondary: #6B7280; --text-muted: #9CA3AF;
    --border: #E5E7EB; --border-hover: #D1D5DB;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.06);
    --shadow-md: 0 4px 20px rgba(0,0,0,0.08);
    --shadow-lg: 0 10px 40px rgba(0,0,0,0.12);
    --radius-sm: 12px; --radius-md: 18px; --radius-lg: 24px; --radius-xl: 32px;
    --sidebar-width: 320px; --header-height: 72px;
    --transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    --vh: 1vh;
}
body.dark {
    --primary: #2EA55B; --primary-light: #3CC96E;
    --primary-glow: rgba(46,165,91,0.12);
    --gold: #D4B85A; --gold-soft: rgba(212,184,90,0.1);
    --bg-page: #0C1017; --bg-card: #151B26; --bg-chat: #0F141D; --bg-input: #1A2030;
    --bubble-user: #1A6B3C; --bubble-user-text: #FFF;
    --bubble-bot: #1E2738; --bubble-bot-text: #E5E7EB;
    --text-primary: #F0F2F5; --text-secondary: #9CA3AF; --text-muted: #6B7280;
    --border: #1E2738; --border-hover: #2A3548;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.2);
    --shadow-md: 0 4px 20px rgba(0,0,0,0.3);
    --shadow-lg: 0 10px 40px rgba(0,0,0,0.4);
}
*,*::before,*::after{box-sizing:border-box}
body,.gradio-container{font-family:'Tajawal',sans-serif!important;background:var(--bg-page)!important;color:var(--text-primary)!important;margin:0;padding:0;overflow-x:hidden;-webkit-text-size-adjust:100%}
footer{display:none!important}
.gradio-container{max-width:100%!important;padding:0!important}
#init-lang-radio .gr-block-label, #init-lang-radio > label:first-child {display:none!important}
#init-lang-radio {text-align:center!important}
#init-lang-radio .gr-radio-row, #init-lang-radio .wrap {justify-content:center!important;display:flex!important;flex-wrap:wrap!important;gap:8px!important}
/* === WELCOME === */
#welcome-container{background:var(--bg-card)!important;border-radius:var(--radius-xl)!important;padding:44px 36px!important;box-shadow:var(--shadow-lg)!important;margin:5vh auto!important;max-width:520px!important;width:calc(100% - 32px)!important;border:1px solid var(--border)!important;text-align:center!important;position:relative;overflow:hidden;transition:opacity .4s ease,transform .4s ease;animation:fadeInUp .5s ease-out}
#welcome-container::before{content:'';position:absolute;top:0;left:50%;transform:translateX(-50%);width:80px;height:4px;background:linear-gradient(90deg,var(--gold),var(--primary));border-radius:0 0 4px 4px}
#welcome-container h1{color:var(--primary)!important;font-weight:800!important;font-size:1.5rem!important;margin-bottom:4px!important}
#welcome-container h3{color:var(--text-secondary)!important;font-weight:400!important;font-size:.95rem!important}
#welcome-container p{color:var(--text-muted)!important;font-size:.88rem!important}
#welcome-container .primary{background:linear-gradient(135deg,var(--primary),var(--primary-light))!important;border:none!important;border-radius:var(--radius-md)!important;padding:14px 40px!important;font-size:1.05rem!important;font-weight:700!important;color:#fff!important;box-shadow:0 4px 16px rgba(13,59,30,.25)!important;transition:var(--transition)!important;margin-top:8px!important;min-height:48px!important}
#welcome-container .primary:hover{transform:translateY(-2px)!important;box-shadow:0 8px 24px rgba(13,59,30,.35)!important}
/* === HEADER === */
#header-container{background:linear-gradient(135deg,var(--primary) 0%,#0A2E17 100%)!important;padding:0 20px!important;margin:0!important;min-height:var(--header-height)!important;border-radius:0!important;border:none!important;box-shadow:0 2px 20px rgba(0,0,0,.15)!important;position:sticky;top:0;z-index:50;align-items:center!important}
#header-container::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--gold),transparent);opacity:.6;animation:pulse-gold 3s ease-in-out infinite}
#header-container > div{align-items:center!important;display:flex!important}
#header-container .gr-column{display:flex!important;align-items:center!important;justify-content:flex-end!important}
.header-btn{min-width:40px!important;max-width:40px!important;height:40px!important;border-radius:12px!important;background:rgba(255,255,255,.08)!important;border:1px solid rgba(255,255,255,.12)!important;color:#fff!important;font-size:1.1rem!important;transition:var(--transition)!important;padding:0!important;margin:0 2px!important;box-shadow:none!important;line-height:40px!important}
.header-btn:hover{background:rgba(255,255,255,.18)!important;transform:scale(1.05);border-color:rgba(255,255,255,.2)!important}
/* === CHAT === */
#chat-workspace{padding:16px 20px 12px!important;max-width:900px;margin:0 auto}
.chatbot-container{background:var(--bg-chat)!important;border:1px solid var(--border)!important;border-radius:var(--radius-lg)!important;box-shadow:var(--shadow-sm)!important;overflow:hidden;height:calc(var(--vh,1vh)*58)!important;min-height:300px!important;max-height:600px!important}
.chatbot-container .wrap{scroll-behavior:smooth!important}

/* ========================================= */
/* [ULTIMATE UI-1 FIX] RTL/LTR Strict Alignment */
/* ========================================= */

/* 1. Force bubble to expand, cancel auto-centering */
.chatbot-container .message{padding:12px 18px!important;border-radius:var(--radius-md)!important;font-size:.95rem!important;line-height:1.7!important;display:block!important;width:fit-content!important;max-width:85%!important;box-shadow:var(--shadow-sm)!important;word-wrap:break-word!important;overflow-wrap:break-word!important;position:relative!important}

/* 2. Convert all inner elements to blocks for alignment */
.chatbot-container .message *{display:block!important}
.chatbot-container .message strong,.chatbot-container .message em,.chatbot-container .message code,.chatbot-container .message span,.chatbot-container .message a{display:inline!important}

/* 3. Strict Arabic (RTL) alignment */
body.is-rtl .chatbot-container .message,body.is-rtl .chatbot-container .message *{direction:rtl!important;text-align:right!important}
body.is-rtl .chatbot-container .message ol,body.is-rtl .chatbot-container .message ul{padding-inline-start:24px!important;padding-inline-end:0!important;margin-right:10px!important;list-style-position:outside!important}
body.is-rtl .chatbot-container .message li{display:list-item!important;margin-bottom:6px!important}

/* 4. Strict English (LTR) alignment */
body.is-ltr .chatbot-container .message,body.is-ltr .chatbot-container .message *{direction:ltr!important;text-align:left!important}
body.is-ltr .chatbot-container .message ol,body.is-ltr .chatbot-container .message ul{padding-inline-start:24px!important;padding-inline-end:0!important;margin-left:10px!important;list-style-position:outside!important}
body.is-ltr .chatbot-container .message li{display:list-item!important;margin-bottom:6px!important}

/* 5. Tables */
.chatbot-container .message table{width:100%!important;display:table!important;border-collapse:collapse!important;margin-top:10px!important}
.chatbot-container .message th,.chatbot-container .message td{display:table-cell!important;border:1px solid var(--border)!important;padding:8px 12px!important}

/* ========================================= */
/* Bubble colors & positioning */
/* ========================================= */
.chatbot-container .user{background:var(--bubble-user)!important;color:var(--bubble-user-text)!important;margin-left:auto!important;margin-right:0!important}
.chatbot-container .bot{background:var(--bubble-bot)!important;color:var(--bubble-bot-text)!important;border:1px solid var(--border)!important;margin-right:auto!important;margin-left:0!important}

/* Dynamic bubble corners */
body.is-rtl .chatbot-container .user{border-radius:var(--radius-md) var(--radius-md) var(--radius-md) 6px!important}
body.is-rtl .chatbot-container .bot{border-radius:var(--radius-md) var(--radius-md) 6px var(--radius-md)!important}
body.is-ltr .chatbot-container .user{border-radius:var(--radius-md) var(--radius-md) 6px var(--radius-md)!important}
body.is-ltr .chatbot-container .bot{border-radius:var(--radius-md) var(--radius-md) var(--radius-md) 6px!important}

/* Copy button — pin to consistent corner */
.chatbot-container .message .copy-btn,.chatbot-container .message button[title="copy"]{position:absolute!important;top:6px!important;inset-inline-end:6px!important;inset-inline-start:auto!important;opacity:.4!important;font-size:.75rem!important;min-width:28px!important;height:28px!important;padding:0!important;border-radius:6px!important}
.chatbot-container .message:hover .copy-btn,.chatbot-container .message:hover button[title="copy"]{opacity:.8!important}
/* === CHIPS === */
.suggestion-row{gap:8px!important;padding:10px 0 6px!important;flex-wrap:wrap!important;justify-content:center!important}
.chip-btn{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:20px!important;padding:8px 16px!important;font-size:.8rem!important;color:var(--text-secondary)!important;cursor:pointer!important;transition:var(--transition)!important;white-space:nowrap!important;font-family:'Tajawal',sans-serif!important;min-height:38px!important}
.chip-btn:hover{background:var(--primary-glow)!important;border-color:var(--primary-light)!important;color:var(--primary)!important;transform:translateY(-1px)!important}
/* === INPUT === */
.modern-input-row{background:var(--bg-input)!important;border-radius:var(--radius-xl)!important;border:1px solid var(--border)!important;box-shadow:var(--shadow-md)!important;align-items:center!important;margin-top:8px!important;transition:var(--transition)!important}
.is-rtl .modern-input-row{padding:6px 14px 6px 8px!important}
.is-ltr .modern-input-row{padding:6px 8px 6px 14px!important}
.modern-input-row:focus-within{border-color:var(--primary-light)!important;box-shadow:var(--shadow-md),0 0 0 3px var(--primary-glow)!important}
#msg-input textarea{border:none!important;background:transparent!important;box-shadow:none!important;resize:none!important;font-size:1rem!important;font-family:'Tajawal',sans-serif!important;color:var(--text-primary)!important;padding:10px 4px!important;text-align:start!important;direction:inherit!important}
#msg-input textarea::placeholder{color:var(--text-muted)!important}
.icon-btn{min-width:44px!important;height:44px!important;border-radius:50%!important;padding:0!important;background:transparent!important;border:none!important;font-size:1.2rem!important;transition:var(--transition)!important;display:flex!important;justify-content:center!important;align-items:center!important;cursor:pointer!important;flex-shrink:0!important}
.icon-btn:hover{background:var(--gold-soft)!important;transform:scale(1.08)}
.send-btn{background:linear-gradient(135deg,var(--primary),var(--primary-light))!important;color:#fff!important;min-width:46px!important;height:46px!important;box-shadow:0 2px 10px rgba(13,59,30,.2)!important}
.send-btn:hover{box-shadow:0 4px 16px rgba(13,59,30,.35)!important;transform:scale(1.08)!important}
/* === VOICE (accordion — Gradio 3.x Audio can't be inlined cleanly) === */
.voice-accordion{margin-top:8px!important}
.voice-accordion .label-wrap{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:var(--radius-md)!important;padding:10px 16px!important;font-size:.85rem!important;color:var(--text-secondary)!important;min-height:44px!important}
.feedback-row{gap:6px!important;padding:4px 0!important;align-items:center!important;justify-content:center!important;flex-wrap:nowrap!important}
.feedback-row > *{flex:none!important}
.feedback-btn{min-width:auto!important;width:auto!important;max-width:60px!important;height:32px!important;border-radius:8px!important;background:var(--bg-card)!important;border:1px solid var(--border)!important;font-size:.95rem!important;transition:var(--transition)!important;cursor:pointer!important;padding:4px 14px!important;line-height:1!important}
.feedback-btn:hover{background:var(--primary-glow)!important;border-color:var(--primary-light)!important;transform:translateY(-1px)!important}
.feedback-btn:disabled,.feedback-btn[disabled]{opacity:.4!important;cursor:not-allowed!important;transform:none!important}
/* === SIDEBAR === */
.sidebar-overlay{position:fixed!important;inset:0;background:rgba(0,0,0,0)!important;z-index:998;pointer-events:none;transition:background .4s ease!important}
.overlay-active{background:rgba(0,0,0,.45)!important;pointer-events:all!important;backdrop-filter:blur(3px);-webkit-backdrop-filter:blur(3px)}
.controls-panel{position:fixed!important;top:0;inset-inline-end:calc(-1*var(--sidebar-width) - 20px);width:var(--sidebar-width)!important;height:100vh!important;height:calc(var(--vh,1vh)*100)!important;background:var(--bg-card)!important;z-index:1000;transition:inset-inline-end .4s cubic-bezier(.4,0,.2,1)!important;box-shadow:-8px 0 30px rgba(0,0,0,.12);padding:28px 24px!important;border-inline-start:1px solid var(--border);overflow-y:auto;-webkit-overflow-scrolling:touch}
.sidebar-open{inset-inline-end:0!important}
.controls-panel h2{color:var(--primary)!important;font-size:1.15rem!important;font-weight:700!important}
.close-btn{border-radius:var(--radius-md)!important;border:1px solid var(--border)!important;color:var(--text-secondary)!important;min-height:44px!important;width:100%!important}
.info-card{background:var(--bg-chat)!important;border:1px solid var(--border)!important;border-radius:var(--radius-sm)!important;padding:14px 16px!important;margin:8px 0!important;direction:ltr!important;text-align:left!important}
.info-card p{margin:4px 0!important;font-size:.85rem!important;color:var(--text-secondary)!important}
.info-card strong{color:var(--text-primary)!important}
/* === RESPONSIVE === */
@media(min-width:1025px){#chat-workspace{padding:20px 32px 16px!important}.chatbot-container{height:520px!important;max-height:620px!important}}
@media(min-width:769px)and(max-width:1024px){#chat-workspace{padding:16px 20px!important}.chatbot-container{height:calc(var(--vh,1vh)*52)!important;max-height:500px!important}.controls-panel{width:300px!important}}
@media(min-width:481px)and(max-width:768px){:root{--header-height:64px;--radius-xl:24px}#welcome-container{margin:4vh 16px!important;padding:36px 28px!important;max-width:480px!important}#welcome-container h1{font-size:1.35rem!important}#chat-workspace{padding:12px 14px!important}.chatbot-container{height:calc(var(--vh,1vh)*50)!important;min-height:280px!important;max-height:450px!important;border-radius:var(--radius-md)!important}.chatbot-container .message{max-width:88%!important;font-size:.92rem!important;padding:10px 14px!important}.controls-panel{width:280px!important}.chip-btn{padding:7px 14px!important;font-size:.78rem!important}.header-brand-kaust{display:none!important}}
@media(max-width:480px){:root{--header-height:60px;--radius-xl:20px;--radius-lg:16px;--radius-md:14px}#welcome-container{margin:2vh 10px!important;padding:28px 20px!important;border-radius:var(--radius-lg)!important;max-width:100%!important}#welcome-container h1{font-size:1.2rem!important}#welcome-container h3{font-size:.85rem!important}#welcome-container .primary{padding:12px 28px!important;font-size:.95rem!important;width:100%!important}#header-container{padding:0 12px!important;min-height:var(--header-height)!important}.header-btn{min-width:36px!important;height:36px!important;border-radius:10px!important;font-size:1rem!important}.header-brand-kaust{display:none!important}#chat-workspace{padding:8px!important}.chatbot-container{height:calc(var(--vh,1vh)*48)!important;min-height:250px!important;max-height:400px!important;border-radius:var(--radius-md)!important}.chatbot-container .message{max-width:92%!important;font-size:.88rem!important;padding:10px 12px!important;line-height:1.6!important}.suggestion-row{flex-wrap:nowrap!important;overflow-x:auto!important;-webkit-overflow-scrolling:touch!important;scrollbar-width:none!important;padding:8px 4px!important;justify-content:flex-start!important;gap:6px!important}.suggestion-row::-webkit-scrollbar{display:none}.chip-btn{padding:7px 12px!important;font-size:.75rem!important;flex-shrink:0!important;min-height:36px!important}.modern-input-row{border-radius:var(--radius-lg)!important;margin-top:6px!important}.is-rtl .modern-input-row{padding:4px 10px 4px 6px!important}.is-ltr .modern-input-row{padding:4px 6px 4px 10px!important}#msg-input textarea{font-size:.92rem!important;padding:8px 4px!important}.icon-btn{min-width:40px!important;height:40px!important;font-size:1.1rem!important}.send-btn{min-width:42px!important;height:42px!important}.controls-panel{width:85vw!important;padding:20px 16px!important}}
@media(max-width:360px){#welcome-container{padding:22px 16px!important}#welcome-container h1{font-size:1.1rem!important}.chatbot-container{height:calc(var(--vh,1vh)*44)!important;min-height:220px!important}.chip-btn{padding:6px 10px!important;font-size:.72rem!important}.controls-panel{width:92vw!important}}
@media(max-height:500px)and(orientation:landscape){.chatbot-container{height:calc(var(--vh,1vh)*40)!important;min-height:180px!important;max-height:250px!important}#welcome-container{margin:2vh auto!important;padding:20px!important}.suggestion-row{display:none!important}}
@supports(padding:env(safe-area-inset-bottom)){.modern-input-row{padding-bottom:calc(6px + env(safe-area-inset-bottom))!important}.controls-panel{padding-bottom:calc(28px + env(safe-area-inset-bottom))!important}}
@keyframes fadeInUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse-gold{0%,100%{opacity:.5}50%{opacity:1}}
@media print{#header-container,.modern-input-row,.suggestion-row,.voice-accordion,.controls-panel,.sidebar-overlay{display:none!important}.chatbot-container{height:auto!important;max-height:none!important}}
"""


# =========================================================================
# 3. HTML
# =========================================================================
HEADER_HTML = """
<div style='display:flex;justify-content:space-between;align-items:center;padding:14px 0;'>
    <div style='display:flex;align-items:center;gap:14px;'>
        <img src='file/ui/assets/saudi_emblem.svg' style='height:40px;filter:brightness(0) invert(1);opacity:.95;' alt='Saudi National Emblem'>
        <div style='border-inline-start:2px solid rgba(200,169,81,.3);padding-inline-start:12px;'>
            <h2 style='color:#fff;margin:0;font-weight:800;font-size:1.15rem;letter-spacing:-.3px;line-height:1.3;'>
                \u0645\u0633\u0627\u0639\u062f \u0623\u0628\u0634\u0631 \u0627\u0644\u0630\u0643\u064a
            </h2>
            <p style='color:rgba(200,169,81,.85);margin:0;font-size:.68rem;font-weight:500;letter-spacing:.4px;'>
                Absher Smart Assistant
            </p>
        </div>
    </div>
    <div style='display:flex;align-items:center;gap:8px;'>
        <span style='background:rgba(200,169,81,.12);color:var(--gold);padding:3px 10px;border-radius:6px;font-size:.6rem;font-weight:600;border:1px solid rgba(200,169,81,.2);letter-spacing:.3px;'>v5.2.0 \u00b7 ALLaM</span>
        <span style='background:rgba(46,165,91,.15);color:#5EE88A;padding:3px 9px;border-radius:6px;font-size:.6rem;font-weight:600;border:1px solid rgba(46,165,91,.15);'>\u25cf Online</span>
        <div class='header-brand-kaust' style='display:flex;align-items:center;gap:6px;border-inline-start:1px solid rgba(255,255,255,.12);padding-inline-start:10px;margin-inline-start:2px;'>
            <img src='file/ui/assets/KAUST.png' style='height:22px;filter:brightness(0) invert(1);opacity:.7;' alt='KAUST Academy'>
            <span style='color:rgba(255,255,255,.5);font-size:.6rem;font-weight:600;'>PGD+</span>
        </div>
    </div>
</div>
"""
SIDEBAR_OVERLAY_HTML = "<div class='sidebar-overlay' onclick='window.toggleMoiSidebar()' role='button' aria-label='Close sidebar'></div>"


# =========================================================================
# 5. GRADIO THEME
# =========================================================================
MOI_THEME = gr.themes.Soft(
    primary_hue="emerald", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Tajawal"), "sans-serif"]
).set(container_radius="18px", button_large_radius="14px", block_label_text_size="0.75rem", block_shadow="none", block_border_width="1px")