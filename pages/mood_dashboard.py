import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from memory import get_all_entries

st.set_page_config(page_title="Mood Dashboard", page_icon="📊")

st.title("📊 Mood Dashboard")
st.caption("Your emotional journey over time.")

entries = get_all_entries()

if not entries:
    st.info("No journal entries yet. Go to the main page and write something!")
else:
    df = pd.DataFrame(entries, columns=[
        'id', 'text', 'sentiment', 'dominant_emotion', 'compound_score', 'timestamp'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    st.subheader("Mood trend over time")
    fig1 = px.line(
        df,
        x='timestamp',
        y='compound_score',
        markers=True,
        color_discrete_sequence=['#1D9E75']
    )
    fig1.update_layout(
        xaxis_title="Date",
        yaxis_title="Mood score (-1 = very negative, +1 = very positive)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=[-1, 1])
    )
    fig1.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Emotion breakdown")
    emotion_counts = df['dominant_emotion'].value_counts().reset_index()
    emotion_counts.columns = ['emotion', 'count']
    fig2 = px.bar(
        emotion_counts,
        x='emotion',
        y='count',
        color='emotion',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Sentiment distribution")
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    fig3 = px.pie(
        sentiment_counts,
        names='sentiment',
        values='count',
        color_discrete_sequence=['#1D9E75', '#D85A30', '#888780']
    )
    fig3.update_layout(
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Recent journal entries")
    for _, row in df.tail(5).iterrows():
        with st.expander(f"{row['timestamp'].strftime('%b %d, %Y')} — {row['sentiment']} ({row['dominant_emotion']})"):
            st.write(row['text'])