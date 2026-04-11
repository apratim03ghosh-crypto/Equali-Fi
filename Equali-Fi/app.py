import streamlit as st
import asyncio
from datetime import datetime
import pandas as pd
import altair as alt

from src.agents.orchestrator import run_consensus
from src.prompts.system_prompts import NEUTRALIZER_PROMPT

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Equali-Fi | AI Governance",
    page_icon="⚖️",
    layout="wide",
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

# --- MODEL AVATARS ---
MODEL_AVATARS = {
    "Gemini 2.0 Flash": "🟡",
    "DeepSeek Chat": "🔵",
    "GPT-4o Mini": "🟢",
    "Mistral Mixtral": "🟣",
    "Google Gemma 2": "⚪"
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

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #11141b;
    border-right: 1px solid #1e293b;
}

/* Chat Animation */
@keyframes fadeSlideUp {
    from {opacity:0; transform:translateY(10px);}
    to {opacity:1; transform:translateY(0);}
}

/* Chat Bubbles */
.user-msg {
    background: linear-gradient(135deg, #6366f1, #4338ca);
    padding: 14px 18px;
    border-radius: 20px 20px 4px 20px;
    margin: 12px 0 12px auto;
    max-width: 75%;
    color: white;
    animation: fadeSlideUp 0.25s ease-out;
}

.asst-msg {
    background: #1e293b;
    padding: 14px 18px;
    border-radius: 20px 20px 20px 4px;
    margin: 12px auto 12px 0;
    max-width: 80%;
    border: 1px solid #334155;
    animation: fadeSlideUp 0.25s ease-out;
}

.time-stamp {
    font-size: 0.65rem;
    color: #94a3b8;
    display: block;
    margin-top: 6px;
    text-align: right;
}

/* Glass Card */
.process-card {
    background: rgba(30, 41, 59, 0.5);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}

/* Winner Badge */
.winner-badge {
    background: rgba(34,197,94,0.2);
    color: #4ade80;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    border: 1px solid #22c55e;
}

/* Sticky Right Panel */
div[data-testid="column"]:nth-child(2) {
    position: sticky;
    top: 20px;
    height: fit-content;
}

/* Input */
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
    "Google Gemma 2": "google/gemma-2-9b-it"
}

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## ⚖️ Equali-Fi")
    st.caption("AI Neutrality Engine")
    st.markdown("---")

    st.caption("NEUTRALITY BOARD")
    selected = st.multiselect(
        "Active Agents",
        list(AVAILABLE_MODELS.keys()),
        default=["Gemini 2.0 Flash"],
        max_selections=3
    )
    active_model_ids = [AVAILABLE_MODELS[x] for x in selected]

    st.markdown("### Conversations")
    if st.button("＋ New Thread", use_container_width=True):
        name = f"Thread {len(st.session_state.chats)+1}"
        st.session_state.chats[name] = [{"role": "system", "content": NEUTRALIZER_PROMPT}]
        st.session_state.current_chat = name
        st.session_state.last_audit = None
        st.session_state.performance_history = []
        st.rerun()

    for chat in st.session_state.chats:
        if st.button(f"💬 {chat}", use_container_width=True):
            st.session_state.current_chat = chat
            st.rerun()

# --- LAYOUT ---
chat_col, analytics_col = st.columns([2.5, 1], gap="large")

# --- CHAT ---
with chat_col:
    st.markdown(f"### {st.session_state.current_chat}")

    messages = st.session_state.chats[st.session_state.current_chat]

    if len(messages) == 1:
        st.markdown("""
        <div style="text-align:center; padding:50px; color:#64748b;">
            <div style="font-size:2.5rem;">⚖️</div>
            <p>Your neutrality engine is ready.</p>
        </div>
        """, unsafe_allow_html=True)

    for msg in messages[1:]:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-msg">
                {msg["content"]}
                <span class="time-stamp">{msg.get("time","")}</span>
            </div>
            """, unsafe_allow_html=True)

        elif msg["role"] == "assistant":
            avatar = MODEL_AVATARS.get(msg.get("model",""), "🤖")
            model_name = msg.get("model", "AI")

            st.markdown(f"""
            <div class="asst-msg">
                <strong>{avatar} {model_name}</strong><br>
                {msg["content"]}
                <span class="time-stamp">{msg.get("time","")}</span>
            </div>
            """, unsafe_allow_html=True)

# --- ANALYTICS PANEL ---
with analytics_col:
    st.markdown("#### ⚙️ Analysis")

    if st.session_state.last_audit:
        audit = st.session_state.last_audit

        st.markdown(f"""
        <div class="process-card">
            <div style="display:flex; justify-content:space-between;">
                <span class="winner-badge">🏆 FINAL VERDICT</span>
                <span>⚖️</span>
            </div>
            <h4 style="margin-top:10px;">{audit['best_ai_name']}</h4>
            <p style="font-size:0.85rem; color:#94a3b8;">
                {audit['reasoning']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.bar_chart(audit["scores"], height=150)

        # --- PERFORMANCE GRAPH ---
        st.markdown("#### 📈 Performance Over Time")

        df = build_performance_df(st.session_state.performance_history)

        if not df.empty:
            df_reset = df.reset_index().melt(id_vars="index")

            chart = alt.Chart(df_reset).mark_line(point=True).encode(
                x="index",
                y="value",
                color="variable"
            ).properties(height=250)

            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("No performance data yet.")

    else:
        st.markdown("""
        <div style="text-align:center; padding:50px; color:#64748b;">
            <div style="font-size:2rem;">⚖️</div>
            <p>Waiting for consensus...</p>
        </div>
        """, unsafe_allow_html=True)

# --- INPUT ---
if prompt := st.chat_input("Ask for a neutral perspective..."):
    if not active_model_ids:
        st.error("Select at least one model.")
    else:
        st.session_state.chats[st.session_state.current_chat].append({
            "role": "user",
            "content": prompt,
            "time": get_time()
        })
        st.rerun()

# --- ARBITRATION ---
messages = st.session_state.chats[st.session_state.current_chat]

if len(messages) > 1 and messages[-1]["role"] == "user":
    with chat_col:
        with st.status("⚖️ Running Neutrality Engine...", expanded=True):
            st.write("🧠 Querying models...")
            st.write("⚡ Generating responses...")
            st.write("📊 Evaluating bias...")
            st.write("🏆 Selecting best answer...")

            try:
                result = asyncio.run(run_consensus(messages, active_model_ids))

                st.session_state.last_audit = result

                # SAVE PERFORMANCE HISTORY
                st.session_state.performance_history.append(result["scores"])

                st.session_state.chats[st.session_state.current_chat].append({
                    "role": "assistant",
                    "content": result["final_answer"],
                    "model": result["best_ai_name"],
                    "time": get_time()
                })

                st.rerun()

            except Exception as e:
                st.error(f"Engine Failure: {e}")