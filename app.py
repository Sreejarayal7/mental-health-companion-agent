import streamlit as st
from dotenv import load_dotenv
import os
from datetime import datetime
from groq import Groq
from utils import analyze
from crisis import check_crisis, get_crisis_message
from memory import save_entry, get_relevant_memories
from anomaly import detect_anomalies, get_anomaly_message
from coping_toolkit import get_coping_techniques, format_techniques_for_display, format_techniques_for_prompt
from therapist_notes import create_session_note
from quotes import get_daily_quote, get_last_mood_from_db

load_dotenv(override=True)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_response(emotion, sentiment, memories, user_input, coping_techniques=""):
    prompt = f"""
You are a warm, empathetic mental health companion. You are NOT a therapist.
Your job is to listen, validate feelings, and suggest healthy coping strategies.
Always recommend professional help for serious issues.

Current emotion detected: {emotion}
Current sentiment: {sentiment}

Relevant past journal memories:
{memories}

Clinically curated coping techniques relevant to this person right now:
{coping_techniques}

User wrote today:
{user_input}

Respond with:
1. Acknowledgement of their feelings (2-3 sentences, warm and personal)
2. Recommend ONE of the coping techniques above — explain it naturally,
   don't just copy it. Make it feel personal to what they shared.
3. A gentle reminder that professional help is always available if needed

Keep your response under 180 words. Be human, not clinical.
"""
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=350,
        stream=True
    )
    return stream

# ── Page config ─────────────────────────────────
st.set_page_config(
    page_title="Mental Health Companion",
    page_icon="💚",
    layout="centered"
)

st.title("💚 Mental Health Companion")
st.caption("A safe space to express how you feel. Your entries are private and stored locally.")

# ── Session state ────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_emotion_data" not in st.session_state:
    st.session_state.last_emotion_data = {}
if "daily_quote" not in st.session_state:
    sentiment, emotion = get_last_mood_from_db()
    st.session_state.daily_quote = get_daily_quote(sentiment, emotion)

# ── Chat history ─────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

techniques_display = []

# ── Chat input ───────────────────────────────────
user_input = st.chat_input("How are you feeling today? Write anything...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    if check_crisis(user_input):
        crisis_msg = "I'm really concerned about you right now. Please reach out immediately:\n\n" + get_crisis_message()
        with st.chat_message("assistant"):
            st.error(crisis_msg)
        st.session_state.messages.append({"role": "assistant", "content": crisis_msg})
    else:
        with st.spinner("Analysing your feelings..."):
            result       = analyze(user_input)
            emotion      = result['dominant_emotion']
            emotion_group = result.get('emotion_group', emotion)
            top_emotions = result.get('top_emotions_str', emotion)
            sentiment    = result['sentiment']['label']
            compound     = result['sentiment']['scores']['compound']

            memories     = get_relevant_memories(user_input)
            memory_text  = "\n".join(memories) if memories else "No previous entries yet."

            save_entry(user_input, sentiment, emotion_group, compound)
            st.session_state.last_emotion_data = result

            # Refresh quote to match new mood
            st.session_state.daily_quote = get_daily_quote(
                sentiment, emotion_group, force_refresh=False
            )

        # ── Anomaly detection ────────────────────
        anomaly_report  = detect_anomalies(compound, sentiment)
        anomaly_message = get_anomaly_message(anomaly_report)

        # ── Coping techniques ────────────────────
        techniques         = get_coping_techniques(emotion, sentiment, user_input)
        techniques_text    = format_techniques_for_prompt(techniques)
        techniques_display = format_techniques_for_display(techniques)

        # ── Get LLM stream ───────────────────────
        stream = get_response(top_emotions, sentiment, memory_text,
                              user_input, techniques_text)

        # ── Show anomaly warning first ───────────
        if anomaly_message:
            with st.chat_message("assistant"):
                st.warning(anomaly_message)
            st.session_state.messages.append({
                "role": "assistant", "content": anomaly_message
            })

        # ── Stream response word by word ─────────
        with st.chat_message("assistant"):
            reply = st.write_stream(
                chunk.choices[0].delta.content or ""
                for chunk in stream
                if chunk.choices[0].delta.content is not None
            )
        st.session_state.messages.append({"role": "assistant", "content": reply})

        # ── Sidebar mood alert ───────────────────
        if anomaly_report["detected"]:
            severity = anomaly_report["severity"]
            icon     = "🔴" if severity == "high" else "🟡"
            st.sidebar.warning(
                f"{icon} **Mood Alert Detected**\n\n"
                + "\n".join([a.split("|")[1] for a in anomaly_report["alerts"]])
            )

        # ── Emotion analysis expander ────────────
        with st.expander("🧠 Emotion Analysis"):
            col1, col2 = st.columns(2)
            with col1:
                sentiment_label = result['sentiment']['label']
                clr = ("#1D9E75" if sentiment_label == "positive"
                       else "#D85A30" if sentiment_label == "negative"
                       else "#888780")
                st.markdown(
                    f"**Sentiment:** <span style='color:{clr}'>"
                    f"{sentiment_label}</span>",
                    unsafe_allow_html=True)
                st.markdown(f"**Mood Score:** `{compound:.2f}`")
                st.markdown(f"**Dominant Emotion:** `{emotion}`")
                st.markdown(f"**Emotion Group:** `{emotion_group}`")
            with col2:
                st.markdown("**Top Emotions:**")
                top5 = list(result.get('emotions', {}).items())[:5]
                for em, score in top5:
                    st.progress(score, text=f"{em}: {score:.1%}")

        # ── Coping toolkit expander ──────────────
        if techniques_display:
            with st.expander("🛠️ Your Personalised Coping Toolkit"):
                st.markdown("*Curated techniques matched to how you're feeling:*")
                for i, tech in enumerate(techniques_display, 1):
                    st.markdown(f"**{i}. {tech['name']}**")
                    st.markdown(f"{tech['desc']}")
                    st.divider()

# ════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════

# ── Daily mood-aware quote ───────────────────────
st.sidebar.markdown("### 💭 Today's Quote")

quote_data    = st.session_state.daily_quote
sentiment_now = st.session_state.last_emotion_data.get(
    'sentiment', {}).get('label', 'neutral')
border_color  = {
    "positive": "#1D9E75",
    "negative": "#7F77DD",
    "neutral":  "#BA7517",
}.get(sentiment_now, "#1D9E75")

st.sidebar.markdown(
    f"""
    <div style="border-left:3px solid {border_color};
                padding:10px 14px;
                border-radius:0 8px 8px 0;
                background:rgba(255,255,255,0.03);
                margin-bottom:8px;">
        <p style="font-size:13px;font-style:italic;
                  color:#e0e0e0;margin:0 0 6px 0;line-height:1.5;">
            "{quote_data['quote']}"
        </p>
        <p style="font-size:11px;color:{border_color};
                  margin:0;font-weight:500;">
            — {quote_data['author']}
        </p>
    </div>
    <p style="font-size:10px;color:#666;margin:0 0 12px 0;">
        ✨ Personalised for your mood
    </p>
    """,
    unsafe_allow_html=True
)

if st.sidebar.button("🔄 New quote", use_container_width=False):
    s, e = get_last_mood_from_db()
    st.session_state.daily_quote = get_daily_quote(s, e, force_refresh=True)
    st.rerun()

st.sidebar.markdown("---")

# ── Session note generator ───────────────────────
st.sidebar.markdown("### 📋 Session Note")
st.sidebar.markdown("Generate a clinical PDF note for this session.")

if st.sidebar.button("📄 Generate Session Note", use_container_width=True):
    if len(st.session_state.messages) < 2:
        st.sidebar.warning("Have a conversation first!")
    else:
        with st.sidebar:
            with st.spinner("Generating your session note..."):
                result_tuple = create_session_note(
                    st.session_state.messages,
                    st.session_state.last_emotion_data
                )
                if result_tuple:
                    pdf_path, note_data = result_tuple
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    st.success("✅ Session note ready!")
                    st.download_button(
                        label="⬇️ Download PDF Note",
                        data=pdf_bytes,
                        file_name=f"session_note_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.markdown(f"**Risk:** {note_data.get('risk_level','low').upper()}")
                    st.markdown(f"**Session:** {note_data.get('session_rating','neutral').upper()}")
                    st.markdown("**Key Themes:**")
                    for theme in note_data.get('key_themes', [])[:3]:
                        st.markdown(f"• {theme}")

st.sidebar.markdown("---")

# ── About ────────────────────────────────────────
st.sidebar.title("About")
st.sidebar.info(
    "This is an AI companion for emotional support. "
    "It is NOT a replacement for professional mental health care. "
    "If you are in crisis, please call iCall: 9152987821"
)