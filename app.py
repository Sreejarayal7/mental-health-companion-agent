import streamlit as st
from dotenv import load_dotenv
import os
from groq import Groq
from utils import analyze
from crisis import check_crisis, get_crisis_message
from memory import save_entry, get_relevant_memories

load_dotenv(override=True)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_response(emotion, sentiment, memories, user_input):
    prompt = f"""
You are a warm, empathetic mental health companion. You are NOT a therapist.
Your job is to listen, validate feelings, and suggest healthy coping strategies.
Always recommend professional help for serious issues.

Current emotion detected: {emotion}
Current sentiment: {sentiment}

Relevant past journal memories:
{memories}

User wrote today:
{user_input}

Respond with:
1. Acknowledgement of their feelings (2-3 sentences, warm and personal)
2. A relevant coping strategy (breathing, journaling, walking, talking to someone)
3. A gentle reminder that professional help is always available if needed

Keep your response under 150 words. Be human, not clinical.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content

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
        with st.spinner("Thinking..."):
            result = analyze(user_input)
            emotion = result['dominant_emotion']
            sentiment = result['sentiment']['label']
            compound = result['sentiment']['scores']['compound']
            memories = get_relevant_memories(user_input)
            memory_text = "\n".join(memories) if memories else "No previous entries yet."
            save_entry(user_input, sentiment, emotion, compound)
            reply = get_response(emotion, sentiment, memory_text, user_input)

        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

st.sidebar.title("About")
st.sidebar.info(
    "This is an AI companion for emotional support. "
    "It is NOT a replacement for professional mental health care. "
    "If you are in crisis, please call iCall: 9152987821"
)