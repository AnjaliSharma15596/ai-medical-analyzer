import streamlit as st
from dotenv import load_dotenv
from analyzer import extract_text_from_pdf, create_vector_store, create_conversation_chain, get_initial_analysis, analyze_report

load_dotenv()

st.set_page_config(
    page_title="MediScan AI",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.markdown("""
<style>
/* Force dark background */
.stApp {
    background-color: #0a0f1e !important;
}
section[data-testid="stAppViewContainer"] {
    background-color: #0a0f1e !important;
}
section[data-testid="stMain"] {
    background-color: #0a0f1e !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0f1e;
    --surface: #111827;
    --surface2: #1a2332;
    --border: #2a3a52;
    --accent: #4af0c8;
    --text: #f0f4f8;
    --muted: #7a8fa6;
    --danger: #ff6b6b;
    --warning: #ffd93d;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.block-container { padding: 2rem 1.5rem !important; max-width: 780px !important; }

.stButton > button {
    background: var(--accent) !important;
    color: #0a0f1e !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 0.65rem 1.5rem !important;
}
.stButton > button:hover {
    background: #6fffd8 !important;
    transform: translateY(-1px) !important;
}

.stTextInput input, .stTextArea textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
}

#MainMenu, footer, header { visibility: hidden; }

.logo { font-family: 'Playfair Display', serif; font-size: 1.8rem; font-weight: 900; color: var(--accent); }
.logo span { color: var(--text); }

.hero-title { font-family: 'Playfair Display', serif; font-size: 2.8rem; font-weight: 900; line-height: 1.15; color: var(--text); margin: 1rem 0 0.5rem; }
.hero-title .hl { color: var(--accent); }
.hero-sub { font-size: 1rem; color: var(--muted); line-height: 1.7; margin-bottom: 2rem; }

.feature-row { display: flex; gap: 1rem; margin: 1.5rem 0; }
.feature-card { flex: 1; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem; text-align: center; }
.feature-icon { font-size: 1.5rem; margin-bottom: 0.4rem; }
.feature-text { font-size: 0.82rem; color: var(--muted); }

.analysis-box { background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--accent); border-radius: 12px; padding: 1.5rem; margin: 1rem 0; line-height: 1.8; font-size: 0.95rem; color: var(--text); }

.chat-user { background: var(--surface2); border: 1px solid var(--border); border-radius: 12px 12px 4px 12px; padding: 1rem 1.2rem; margin: 0.8rem 0; font-size: 0.92rem; color: var(--text); }
.chat-ai { background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 4px 12px 12px 12px; padding: 1rem 1.2rem; margin: 0.8rem 0; font-size: 0.92rem; line-height: 1.7; color: var(--text); }

.warning-box { background: rgba(255,107,107,0.08); border: 1px solid rgba(255,107,107,0.2); border-radius: 10px; padding: 0.8rem 1rem; font-size: 0.85rem; color: #ff9999; margin: 1rem 0; }
.divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
.sec-btn > button { background: var(--surface2) !important; color: var(--text) !important; border: 1px solid var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# Session state
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "initial_analysis" not in st.session_state:
    st.session_state.initial_analysis = None
if "report_uploaded" not in st.session_state:
    st.session_state.report_uploaded = False
if "report_name" not in st.session_state:
    st.session_state.report_name = ""


def go_home():
    for key in ["conversation_chain", "chat_history", "initial_analysis", "report_uploaded", "report_name"]:
        st.session_state[key] = None if key != "chat_history" else []
    st.session_state.report_uploaded = False
    st.session_state.report_name = ""
    st.rerun()


# ── HOME PAGE ─────────────────────────────────────────────────────────────────
if not st.session_state.report_uploaded:

    st.markdown('<div class="logo">Medi<span>Scan AI</span></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero-title">Understand your<br><span class="hl">medical reports</span><br>instantly.</div>
    <div class="hero-sub">Upload any medical report PDF — blood test, lab results, health checkup.<br>Get instant AI-powered explanation in simple language.</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-row">
        <div class="feature-card"><div class="feature-icon">📋</div><div class="feature-text">Instant Report Summary</div></div>
        <div class="feature-card"><div class="feature-icon">⚠️</div><div class="feature-text">Abnormal Value Detection</div></div>
        <div class="feature-card"><div class="feature-icon">💬</div><div class="feature-text">Ask Follow-up Questions</div></div>
        <div class="feature-card"><div class="feature-icon">🔒</div><div class="feature-text">Private & Secure</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("#### 📁 Upload Your Medical Report")

    uploaded_file = st.file_uploader(
        "Drag & drop your PDF here",
        type=["pdf"],
        help="Supported: Blood reports, Lab results, Health checkup reports"
    )

    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>Disclaimer:</strong> MediScan AI is for educational purposes only.
        Always consult a qualified doctor for medical advice and diagnosis.
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file is not None:
        if st.button("🔍 Analyze Report", use_container_width=True):
            with st.spinner("📖 Reading your report..."):
                report_text = extract_text_from_pdf(uploaded_file)

            if "Error" in report_text or len(report_text.strip()) < 50:
                st.error("Could not extract text. Please try a different PDF file.")
            else:
                with st.spinner("🧠 Creating AI knowledge base..."):
                    vector_store = create_vector_store(report_text)
                    chain = create_conversation_chain(vector_store)
                    st.session_state.conversation_chain = chain

                with st.spinner("🔍 Analyzing your report... please wait"):
                    initial_analysis = get_initial_analysis(chain)
                    st.session_state.initial_analysis = initial_analysis
                    st.session_state.report_uploaded = True
                    st.session_state.report_name = uploaded_file.name
                st.rerun()


# ── ANALYSIS PAGE ─────────────────────────────────────────────────────────────
else:
    st.markdown('<div class="logo">Medi<span>Scan AI</span></div>', unsafe_allow_html=True)
    st.markdown(f"**📄 Report:** {st.session_state.report_name}")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown("### 🔍 AI Analysis")
    st.markdown(f'<div class="analysis-box">{st.session_state.initial_analysis}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if st.session_state.chat_history:
        st.markdown("### 💬 Your Questions")
        for chat in st.session_state.chat_history:
            st.markdown(f'<div class="chat-user">🙋 {chat["question"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-ai">🤖 {chat["answer"]}</div>', unsafe_allow_html=True)

    st.markdown("### ❓ Ask a Question")
    question = st.text_input("", placeholder="Example: What does my hemoglobin level mean? Is my sugar level normal?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Ask AI", use_container_width=True):
            if question.strip():
                with st.spinner("🤖 Thinking..."):
                    answer = analyze_report(st.session_state.conversation_chain, question)
                    st.session_state.chat_history.append({"question": question, "answer": answer})
                st.rerun()
            else:
                st.warning("Please type a question first!")

    with col2:
        st.markdown('<div class="sec-btn">', unsafe_allow_html=True)
        if st.button("📁 Upload New Report", use_container_width=True):
            go_home()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("#### 💡 Suggested Questions")

    suggestions = [
        "Are all my values within normal range?",
        "Which values need immediate attention?",
        "What lifestyle changes should I make?",
        "Should I be worried about any findings?"
    ]

    col1, col2 = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown('<div class="sec-btn">', unsafe_allow_html=True)
            if st.button(f"💬 {suggestion}", use_container_width=True, key=f"sug_{i}"):
                with st.spinner("🤖 Thinking..."):
                    answer = analyze_report(st.session_state.conversation_chain, suggestion)
                    st.session_state.chat_history.append({"question": suggestion, "answer": answer})
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box" style="margin-top:2rem">
        ⚠️ <strong>Remember:</strong> This AI analysis is for information only.
        Please consult your doctor for proper medical advice.
    </div>
    """, unsafe_allow_html=True)
