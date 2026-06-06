import streamlit as st
from dotenv import load_dotenv
import os
from groq import Groq
from utils import analyze
from crisis import check_crisis, get_crisis_message
from memory import save_entry, get_relevant_memories
from anomaly import detect_anomalies, get_anomaly_message
from coping_toolkit import get_coping_techniques, format_techniques_for_display, format_techniques_for_prompt

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

st.set_page_config(
    page_title="Mental Health Companion",
    page_icon="💚",
    layout="centered"
)

st.title("💚 Mental Health Companion")
st.caption("A safe space to express how you feel. Your entries are private and stored locally.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

techniques_display = []

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
            result = analyze(user_input)
            emotion = result['dominant_emotion']
            emotion_group = result.get('emotion_group', emotion)
            top_emotions = result.get('top_emotions_str', emotion)
            sentiment = result['sentiment']['label']
            compound = result['sentiment']['scores']['compound']

            memories = get_relevant_memories(user_input)
            memory_text = "\n".join(memories) if memories else "No previous entries yet."

            save_entry(user_input, sentiment, emotion_group, compound)

        # Anomaly detection
        anomaly_report  = detect_anomalies(compound, sentiment)
        anomaly_message = get_anomaly_message(anomaly_report)

        # Fetch personalised coping techniques from RAG knowledge base
        techniques      = get_coping_techniques(emotion, sentiment, user_input)
        techniques_text = format_techniques_for_prompt(techniques)
        techniques_display = format_techniques_for_display(techniques)

        stream = get_response(top_emotions, sentiment, memory_text,
                              user_input, techniques_text)

        # Stream response word by word — exactly like ChatGPT
        # Show anomaly warning FIRST if detected
        if anomaly_message:
            with st.chat_message("assistant"):
                st.warning(anomaly_message)
            st.session_state.messages.append({
                "role": "assistant", "content": anomaly_message
            })

        # Stream normal response word by word
        with st.chat_message("assistant"):
            reply = st.write_stream(
                chunk.choices[0].delta.content or ""
                for chunk in stream
                if chunk.choices[0].delta.content is not None
            )
        st.session_state.messages.append({"role": "assistant", "content": reply})

        # Show anomaly badge in sidebar if detected
        if anomaly_report["detected"]:
            severity = anomaly_report["severity"]
            color    = "🔴" if severity == "high" else "🟡"
            st.sidebar.warning(
                f"{color} **Mood Alert Detected**\n\n"
                + "\n".join([a.split("|")[1] for a in anomaly_report["alerts"]])
            )

        # Show emotion analysis in expander
        with st.expander("🧠 Emotion Analysis"):
            col1, col2 = st.columns(2)
            with col1:
                compound = result['sentiment']['scores']['compound']
                sentiment_label = result['sentiment']['label']
                color = "#1D9E75" if sentiment_label == "positive" else "#D85A30" if sentiment_label == "negative" else "#888780"
                st.markdown(f"**Sentiment:** <span style='color:{color}'>{sentiment_label}</span>", unsafe_allow_html=True)
                st.markdown(f"**Mood Score:** `{compound:.2f}`")
                st.markdown(f"**Dominant Emotion:** `{emotion}`")
                st.markdown(f"**Emotion Group:** `{emotion_group}`")
            with col2:
                st.markdown("**Top Emotions:**")
                emotions = result.get('emotions', {})
                top5 = list(emotions.items())[:5]
                for em, score in top5:
                    st.progress(score, text=f"{em}: {score:.1%}")

st.sidebar.title("About")
st.sidebar.info(
    "This is an AI companion for emotional support. "
    "It is NOT a replacement for professional mental health care. "
    "If you are in crisis, please call iCall: 9152987821"
)
# Show coping toolkit
if techniques_display:
    with st.expander("🛠️ Your Personalised Coping Toolkit"):
        st.markdown("*Curated techniques matched to how you're feeling right now:*")
        for i, tech in enumerate(techniques_display, 1):
            st.markdown(f"**{i}. {tech['name']}**")
            st.markdown(f"{tech['desc']}")
            st.divider()