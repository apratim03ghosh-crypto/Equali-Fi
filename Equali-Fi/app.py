import streamlit as st
import asyncio
from datetime import datetime
import html
import pandas as pd
import re
import altair as alt

from src.agents.orchestrator import run_consensus
from src.prompts.system_prompts import NEUTRALIZER_PROMPT

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Equali-Fi | AI Governance",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- HELPERS ---
def get_time():
    return datetime.now().strftime("%I:%M %p")

def build_performance_df(history):
    if not history:
        return pd.DataFrame()
    df = pd.DataFrame(history)
    df.index = [f"Q{i+1}" for i in range(len(df))]
    return df

def normalize_display_text(text, fallback="No data available"):
    if not text:
        return fallback

    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    cleaned = text.strip()
    return cleaned or fallback

def can_delete_chat(chat_name):
    return True

def get_chat_message_count(chat_name):
    messages = st.session_state.chats.get(chat_name, [])
    return max(len(messages) - 1, 0)

# --- MODEL AVATARS ---
MODEL_AVATARS = {
    "Gemini 2.0 Flash": "🟡",
    "DeepSeek Chat": "🔵",
    "GPT-4o Mini": "🟢",
    "Mistral Mixtral": "🟣",
    "Google Gemma 4": "⚪",
    "Llama 3.3 70B": "🦙",
    "Qwen 2.5 72B": "🟠",
}

# --- CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
    background-color: #0b0e14;
    color: #e2e8f0;
}

body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background:
        radial-gradient(circle at 18% 20%, rgba(255, 255, 255, 0.035), transparent 0 12%, rgba(255,255,255,0.012) 12.5%, transparent 13.5%),
        radial-gradient(circle at 74% 14%, rgba(255, 255, 255, 0.03), transparent 0 10%, rgba(255,255,255,0.012) 10.5%, transparent 11.5%),
        radial-gradient(circle at 52% 62%, rgba(255, 255, 255, 0.02), transparent 0 18%, rgba(255,255,255,0.008) 18.3%, transparent 19.1%),
        radial-gradient(1px 1px at 7% 13%, rgba(255,255,255,0.78) 45%, transparent 55%),
        radial-gradient(1px 1px at 16% 78%, rgba(255,255,255,0.54) 45%, transparent 55%),
        radial-gradient(1px 1px at 28% 44%, rgba(255,255,255,0.64) 45%, transparent 55%),
        radial-gradient(1px 1px at 41% 23%, rgba(255,255,255,0.52) 45%, transparent 55%),
        radial-gradient(1px 1px at 57% 22%, rgba(255,255,255,0.7) 45%, transparent 55%),
        radial-gradient(1px 1px at 69% 67%, rgba(255,255,255,0.52) 45%, transparent 55%),
        radial-gradient(1px 1px at 83% 31%, rgba(255,255,255,0.62) 45%, transparent 55%),
        radial-gradient(1px 1px at 92% 74%, rgba(255,255,255,0.7) 45%, transparent 55%),
        linear-gradient(180deg, #05070a 0%, #06080d 55%, #05070a 100%);
}

[data-testid="stAppViewContainer"] {
    max-width: none !important;
}

[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] > .main > div,
[data-testid="stAppViewContainer"] > .main > div > div {
    max-width: none !important;
    width: 100% !important;
}

[data-testid="stSidebar"] {
    background: rgba(9, 12, 18, 0.9);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.1rem;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.4rem;
}

[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    letter-spacing: -0.02em;
}

[data-testid="stSidebar"] h2 {
    display: none;
}

.sidebar-brand {
    padding: 0.1rem 0 0.7rem 0;
}

.sidebar-brand__eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.18rem 0.58rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #94a3b8;
    font-size: 0.67rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

.sidebar-brand__title {
    margin: 0.8rem 0 0.12rem 0;
    font-size: 1.5rem;
    font-weight: 650;
    color: #f8fafc;
    letter-spacing: -0.03em;
}

.sidebar-brand__subtitle {
    color: #8a96a8;
    font-size: 0.9rem;
    line-height: 1.45;
    margin-bottom: 0.7rem;
}

.sidebar-divider {
    height: 1px;
    margin: 0.7rem 0 0.95rem;
    background: rgba(255, 255, 255, 0.06);
}

.sidebar-section-label {
    color: #7c8799;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 0.15rem 0 0.3rem;
}

.sidebar-section-title {
    color: #f8fafc;
    font-size: 0.98rem;
    font-weight: 620;
    letter-spacing: -0.02em;
    margin: 0 0 0.55rem;
}

[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p {
    color: #cbd5e1 !important;
    font-size: 0.84rem !important;
    font-weight: 550 !important;
}

[data-testid="stSidebar"] [data-testid="stMultiSelect"] > div > div {
    min-height: 2.8rem;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.02);
    box-shadow: none;
}

[data-testid="stSidebar"] [data-baseweb="tag"] {
    border-radius: 999px !important;
    background: rgba(55, 65, 81, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
}

[data-testid="stSidebar"] [data-baseweb="tag"] span {
    color: #e5e7eb !important;
}

[data-testid="stSidebar"] .stButton > button {
    border-radius: 12px;
    min-height: 2.55rem;
    border: 1px solid rgba(255, 255, 255, 0.07);
    background: rgba(255, 255, 255, 0.02);
    color: #e5e7eb;
    font-weight: 550;
    transition: all 0.18s ease;
    box-shadow: none;
    text-align: left;
    justify-content: flex-start;
    padding-left: 0.8rem;
}

[data-testid="stSidebar"] .stButton > button:hover {
    border-color: rgba(255, 255, 255, 0.11);
    background: rgba(255, 255, 255, 0.045);
    transform: none;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(255, 255, 255, 0.11);
    color: #ffffff;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: rgba(255, 255, 255, 0.07);
}

[data-testid="stSidebar"] [data-testid="column"]:nth-child(2) .stButton > button {
    justify-content: center;
    text-align: center;
    padding-left: 0;
    min-height: 2.85rem;
}

.conversation-meta {
    color: #6b7280;
    font-size: 0.7rem;
    margin: -0.05rem 0 0.32rem 0.15rem;
}

.conversation-list-spacer {
    height: 0.15rem;
}

@media (max-width: 900px) {
    [data-testid="stSidebar"] {
        min-width: 100vw !important;
        max-width: 100vw !important;
    }
}

@keyframes fadeSlideUp {
    from {opacity:0; transform:translateY(10px);}
    to {opacity:1; transform:translateY(0);}
}

.user-msg {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.92), rgba(67, 56, 202, 0.88));
    padding: 14px 18px;
    border-radius: 20px 20px 4px 20px;
    margin: 12px 0 12px auto;
    width: auto;
    max-width: 80%;
    box-sizing: border-box;
    color: white;
    animation: fadeSlideUp 0.25s ease-out;
    border: 1px solid rgba(129, 140, 248, 0.35);
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.32);
    backdrop-filter: blur(10px);
}

.asst-msg {
    background: rgba(15, 23, 42, 0.74);
    padding: 14px 18px;
    border-radius: 20px 20px 20px 4px;
    margin: 12px auto 12px 0;
    width: auto;
    max-width: 80%;
    box-sizing: border-box;
    border: 1px solid rgba(148, 163, 184, 0.18);
    animation: fadeSlideUp 0.25s ease-out;
    backdrop-filter: blur(12px);
    box-shadow: 0 18px 40px rgba(2, 6, 23, 0.22);
}

.time-stamp {
    font-size: 0.65rem;
    color: #94a3b8;
    display: block;
    margin-top: 6px;
    text-align: right;
}

.process-card {
    background: rgba(15, 23, 42, 0.58);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 20px 40px rgba(2, 6, 23, 0.18);
}

.winner-badge {
    background: rgba(34,197,94,0.2);
    color: #4ade80;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    border: 1px solid #22c55e;
}

div[data-testid="column"]:nth-child(2) {
    position: sticky;
    top: 20px;
    height: fit-content;
}

textarea {
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    background-color: #0f172a !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "chats" not in st.session_state:
    st.session_state.chats = {
        "Default Chat": [{"role": "system", "content": NEUTRALIZER_PROMPT}]
    }

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Default Chat"

if "last_audit" not in st.session_state:
    st.session_state.last_audit = None

if "performance_history" not in st.session_state:
    st.session_state.performance_history = []

# --- MODELS ---
AVAILABLE_MODELS = {
    "Gemini 2.0 Flash": "google/gemini-2.0-flash-001",
    "DeepSeek Chat": "deepseek/deepseek-chat",
    "GPT-4o Mini": "openai/gpt-4o-mini",
    "Mistral Mixtral": "mistralai/mixtral-8x7b-instruct",
    "Google Gemma 4": "google/gemma-4-31b-it",
    "Llama 3.3 70B": "meta-llama/llama-3.3-70b-instruct",
    "Qwen 2.5 72B": "qwen/qwen-2.5-72b-instruct",
}

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand__eyebrow">⚖ Governance Workspace</div>
        <div class="sidebar-brand__title"> ⚖️ Equali-Fi</div>
        <div class="sidebar-brand__subtitle">
            Compare model perspectives, manage threads clearly, and arrive at a stronger final answer.
        </div>
    </div>
    <div class="sidebar-divider"></div>
    """, unsafe_allow_html=True)
    st.markdown("##  ⚖️   Equali-Fi")
    st.markdown('<div class="sidebar-section-label">Neutrality Board</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">Active Agents</div>', unsafe_allow_html=True)
    selected = st.multiselect(
        "Active Agents",
        list(AVAILABLE_MODELS.keys()),
        default=["Gemini 2.0 Flash"],
        max_selections=3,
        label_visibility="collapsed",
    )
    active_model_ids = [AVAILABLE_MODELS[x] for x in selected]

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Conversations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">Threads</div>', unsafe_allow_html=True)
    if st.button("＋ New Thread", width='stretch'):
        name = f"Thread {len(st.session_state.chats)+1}"
        st.session_state.chats[name] = [{"role": "system", "content": NEUTRALIZER_PROMPT}]
        st.session_state.current_chat = name
        st.session_state.last_audit = None
        st.session_state.performance_history = []
        st.rerun()

    for chat in list(st.session_state.chats.keys()):
        is_active = st.session_state.current_chat == chat
        message_count = get_chat_message_count(chat)
        turn_label = "Current thread" if is_active else ("Empty" if message_count == 0 else f"{message_count} messages")
        st.markdown(f'<div class="conversation-meta">{turn_label}</div>', unsafe_allow_html=True)
        chat_col_btn, chat_col_delete = st.columns([6, 1], gap="small")

        with chat_col_btn:
            if st.button(
                f"Chat  {chat}",
                key=f"thread_{chat}",
                width='stretch',
                type="primary" if is_active else "secondary",
            ):
                st.session_state.current_chat = chat
                st.rerun()

        with chat_col_delete:
            if st.button("X", key=f"remove_{chat}", width='stretch', help="Delete this thread"):
                del st.session_state.chats[chat]

                if not st.session_state.chats:
                    st.session_state.chats["New Thread"] = [{"role": "system", "content": NEUTRALIZER_PROMPT}]
                    st.session_state.current_chat = "New Thread"
                    st.session_state.last_audit = None
                    st.session_state.performance_history = []
                elif st.session_state.current_chat == chat:
                    st.session_state.current_chat = next(iter(st.session_state.chats))
                    st.session_state.last_audit = None
                    st.session_state.performance_history = []

                st.rerun()

        st.markdown('<div class="conversation-list-spacer"></div>', unsafe_allow_html=True)

    for chat in []:
        is_active = st.session_state.current_chat == chat
        message_count = get_chat_message_count(chat)
        turn_label = "Current thread" if is_active else ("Empty" if message_count == 0 else f"{message_count} messages")
        st.markdown(f'<div class="conversation-meta">{turn_label}</div>', unsafe_allow_html=True)
        chat_col_btn, chat_col_delete = st.columns([6, 1], gap="small")

        with chat_col_btn:
            if st.button(f"💬 {chat}", key=f"open_{chat}", width='stretch'):
                st.session_state.current_chat = chat
                st.rerun()

        with chat_col_delete:
            delete_disabled = not can_delete_chat(chat)
            delete_help = "Delete this thread"

            if st.button("✕", key=f"delete_{chat}", width='stretch', disabled=delete_disabled, help=delete_help):
                del st.session_state.chats[chat]

                if not st.session_state.chats:
                    st.session_state.chats["New Thread"] = [{"role": "system", "content": NEUTRALIZER_PROMPT}]
                    st.session_state.current_chat = "New Thread"
                    st.session_state.last_audit = None
                    st.session_state.performance_history = []
                elif st.session_state.current_chat == chat:
                    st.session_state.current_chat = next(iter(st.session_state.chats))
                    st.session_state.last_audit = None
                    st.session_state.performance_history = []

                st.rerun()

    for chat in []:
        if st.button(f"💬 {chat}", width='stretch'):
            st.session_state.current_chat = chat
            st.rerun()

# --- LAYOUT ---
chat_col, analytics_col = st.columns([3, 1], gap="large")

# --- CHAT ---
with chat_col:
    st.markdown(f"### {st.session_state.current_chat}")

    messages = st.session_state.chats[st.session_state.current_chat]

    # --- EMPTY STATE ---
    if len(messages) == 1:
        st.markdown("""
        <div style="text-align:center; padding:50px; color:#64748b;">
            <div style="font-size:2.5rem;">⚖️</div>
            <p>Your neutrality engine is ready.</p>
        </div>
        """, unsafe_allow_html=True)

    # --- CHAT LOOP ---
    for msg in messages[1:]:

        # --- USER MESSAGE ---
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-msg">
                {msg["content"]}
                <span class="time-stamp">{msg.get("time","")}</span>
            </div>
            """, unsafe_allow_html=True)

        # --- ASSISTANT MESSAGE ---
        elif msg["role"] == "assistant":
            avatar = MODEL_AVATARS.get(msg.get("model",""), "🤖")
            model_name = msg.get("model", "AI")
            safe_final_answer = html.escape(normalize_display_text(msg.get("content"))).replace("\n", "<br>")

            # --- FINAL ANSWER ---
            st.markdown(f"""
            <div class="asst-msg">
                <strong>{avatar} {model_name} (Final Answer)</strong><br>
                {safe_final_answer}
                <span class="time-stamp">{msg.get("time","")}</span>
            </div>
            """, unsafe_allow_html=True)

            # --- 🔥 ENHANCED DEBATE MODE ---
            if st.session_state.last_audit and "responses" in st.session_state.last_audit:

                audit = st.session_state.last_audit

                with st.expander("🔍 View AI Debate", expanded=False):

                    # --- 🔥 KEY DIFFERENCES (CONTEXT FIRST) ---
                    if audit.get("key_differences"):
                        safe_key_diff = html.escape(normalize_display_text(audit.get("key_differences"))).replace("\n", "<br>")
                        st.markdown(f"""
                        <div class="process-card">
                            <strong>🔍 Key Differences</strong>
                            <hr style="border:0.5px solid #334155; margin:6px 0;">
                            <p style="font-size:0.85rem; color:#cbd5f5;">
                                {safe_key_diff}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # --- MODEL RESPONSES ---
                    for model, response in audit["responses"].items():
                        avatar = MODEL_AVATARS.get(model, "🤖")

                        safe_response = html.escape(normalize_display_text(response)).replace("\n", "<br>")

                        st.markdown(f"""
                        <div class="process-card">
                            <strong>{avatar} {model}</strong>
                            <hr style="border:0.5px solid #334155; margin:6px 0;">
                            <p style="font-size:0.85rem;">
                                {safe_response}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

# --- ANALYTICS ---
with analytics_col:
    st.markdown("#### ⚙️ Analysis")

    if st.session_state.last_audit:
        audit = st.session_state.last_audit

        # --- 🔥 CLEAN TEXT FUNCTION ---
        clean_reasoning = normalize_display_text(audit.get("reasoning"))
        clean_diff = normalize_display_text(audit.get("key_differences"), "No comparison available")

        # --- VERDICT CARD ---
        reasoning_html = html.escape(clean_reasoning).replace("\n", "<br>")
        diff_html = html.escape(clean_diff).replace("\n", "<br>")
        st.markdown(f"""
        <div class="process-card">
            <span class="winner-badge">🏆 FINAL VERDICT</span>
            <h4>{audit['best_ai_name']}</h4>

            <div style="font-size:0.85rem; color:#94a3b8; margin-top:6px;">
                <strong>Why this won:</strong><br>
                {reasoning_html}
            </div>

            <hr style="border:0.5px solid #334155; margin:10px 0;">

            <div style="font-size:0.85rem; color:#cbd5f5;">
                <strong>🔍 Key Differences:</strong><br>
                {diff_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- SCORES ---
        st.bar_chart(audit["scores"], height=150)

        # --- 🏆 LEADERBOARD ---
        st.markdown("#### 🏆 Model Rankings")

        def compute_win_rates(history):
            if not history:
                return {}

            win_counts = {}
            total = len(history)

            for entry in history:
                winner = max(entry, key=entry.get)
                win_counts[winner] = win_counts.get(winner, 0) + 1

            return {
                model: round((count / total) * 100, 1)
                for model, count in sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
            }

        win_rates = compute_win_rates(st.session_state.performance_history)

        if win_rates:
            for i, (model, rate) in enumerate(win_rates.items(), start=1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "•"

                st.markdown(f"""
                <div style="
                    display:flex;
                    justify-content:space-between;
                    padding:8px 12px;
                    border-radius:10px;
                    background:#111827;
                    margin-bottom:6px;
                    border:1px solid #1f2937;
                ">
                    <span>{medal} {model}</span>
                    <span>{rate}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No ranking data yet.")

        # --- 📈 PERFORMANCE GRAPH ---
        st.markdown("#### 📈 Performance Over Time")

        df = build_performance_df(st.session_state.performance_history)

        if not df.empty:
            df = df.reset_index().melt(id_vars="index")
            df.rename(columns={
                "index": "Query",
                "value": "Score",
                "variable": "Model"
            }, inplace=True)

            chart = alt.Chart(df).mark_line(point=True).encode(
                x="Query",
                y="Score",
                color="Model"
            ).properties(height=250)

            st.altair_chart(chart, width='stretch')
        else:
            st.caption("No performance data yet.")

    else:
        st.markdown("""
        <div style="text-align:center; padding:50px; color:#64748b;">
            <div style="font-size:2rem;">⚖️</div>
            <p>Waiting for consensus...</p>
        </div>
        """, unsafe_allow_html=True)

# --- INPUT PROCESSING ---
if prompt := st.chat_input("Enter your query..."):
    # 1. Append user message to history
    user_msg = {"role": "user", "content": prompt, "time": get_time()}
    st.session_state.chats[st.session_state.current_chat].append(user_msg)
    
    # 2. Run the engine within a single status block
    with st.status("⚖️ Running Neutrality Engine...", expanded=True) as status:
        # st.write("🧠 Querying multiple AI agents...")
        # st.write("🗣️ Generating independent responses...")
        # st.write("⚖️ Running neutrality evaluation...")
        # st.write("🏆 Selecting best answer...")

        try:
            # Execute consensus logic
            result = asyncio.run(run_consensus(st.session_state.chats[st.session_state.current_chat], active_model_ids))
            
            # 3. Update state with results
            result["last_query"] = prompt
            st.session_state.last_audit = result
            st.session_state.performance_history.append(result["scores"])

            # 4. Append assistant response to history
            st.session_state.chats[st.session_state.current_chat].append({
                "role": "assistant",
                "content": result["final_answer"],
                "model": result["best_ai_name"],
                "time": get_time()
            })
            
            st.write("🏆 Best answer selected.")
            status.update(label="Analysis Complete!", state="complete", expanded=False)
            
            # 5. Refresh to show new messages
            st.rerun()

        except Exception as e:
            status.update(label="Engine Failure", state="error")
            st.error(f"Engine Failure: {e}")
