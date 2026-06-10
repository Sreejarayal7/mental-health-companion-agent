import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from memory import get_all_entries

st.set_page_config(page_title="Mood Dashboard", page_icon="📊")

st.title("📊 Mood Dashboard")
st.caption("Your emotional journey over time.")

user_id = st.session_state.get("user_id", 0)
entries = get_all_entries(user_id=user_id)

if not entries:
    st.info("No journal entries yet. Go to the main page and write something!")
else:
    # ── Build DataFrame ──────────────────────────
    df = pd.DataFrame(entries, columns=[
    'id', 'text', 'sentiment', 'dominant_emotion',
    'compound_score', 'timestamp', 'user_id'
])
    df['timestamp']     = pd.to_datetime(df['timestamp'])
    df['compound_score'] = df['compound_score'].astype(float)
    df = df.sort_values('timestamp')

    # ── Date range filter ────────────────────────
    st.markdown("### 🗓️ Filter by time period")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        filter_option = st.selectbox(
            "Show entries from:",
            ["Last 7 days", "Last 30 days",
             "Last 90 days", "All time", "Custom range"],
            index=1  # Default to last 30 days
        )

    # Apply filter
    now = datetime.now()
    if filter_option == "Last 7 days":
        start_date = now - timedelta(days=7)
        filtered_df = df[df['timestamp'] >= start_date]
    elif filter_option == "Last 30 days":
        start_date = now - timedelta(days=30)
        filtered_df = df[df['timestamp'] >= start_date]
    elif filter_option == "Last 90 days":
        start_date = now - timedelta(days=90)
        filtered_df = df[df['timestamp'] >= start_date]
    elif filter_option == "Custom range":
        with col2:
            start = st.date_input("From", value=df['timestamp'].min().date())
        with col3:
            end = st.date_input("To", value=now.date())
        filtered_df = df[
            (df['timestamp'].dt.date >= start) &
            (df['timestamp'].dt.date <= end)
        ]
    else:  # All time
        filtered_df = df.copy()

    # ── Stats summary row ────────────────────────
    st.markdown("---")
    if filtered_df.empty:
        st.warning("No entries found for this time period. Try a wider range!")
    else:
        # Summary metrics
        total    = len(filtered_df)
        avg_mood = filtered_df['compound_score'].mean()
        positive = len(filtered_df[filtered_df['sentiment'] == 'positive'])
        negative = len(filtered_df[filtered_df['sentiment'] == 'negative'])
        neutral  = len(filtered_df[filtered_df['sentiment'] == 'neutral'])

        # Mood trend direction
        if len(filtered_df) >= 2:
            first_half = filtered_df.head(len(filtered_df)//2)['compound_score'].mean()
            second_half = filtered_df.tail(len(filtered_df)//2)['compound_score'].mean()
            trend = "📈 Improving" if second_half > first_half else "📉 Declining" if second_half < first_half else "➡️ Stable"
        else:
            trend = "➡️ Not enough data"

        # Display metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total entries", total)
        m2.metric("Avg mood", f"{avg_mood:.2f}")
        m3.metric("😊 Positive", positive)
        m4.metric("😔 Negative", negative)
        m5.metric("Trend", trend)

        st.markdown("---")

        # ── Chart 1: Mood trend line ─────────────
        st.subheader("Mood trend over time")

        # Add 7-day rolling average if enough data
        if len(filtered_df) >= 7:
            filtered_df = filtered_df.copy()
            filtered_df['rolling_avg'] = (
                filtered_df['compound_score']
                .rolling(window=min(7, len(filtered_df)), min_periods=1)
                .mean()
            )
            fig1 = go.Figure()
            # Raw scores
            fig1.add_trace(go.Scatter(
                x=filtered_df['timestamp'],
                y=filtered_df['compound_score'],
                mode='markers+lines',
                name='Daily mood',
                line=dict(color='#1D9E75', width=1.5),
                marker=dict(size=6),
                opacity=0.6
            ))
            # Rolling average
            fig1.add_trace(go.Scatter(
                x=filtered_df['timestamp'],
                y=filtered_df['rolling_avg'],
                mode='lines',
                name='7-day average',
                line=dict(color='#7F77DD', width=2.5)
            ))
        else:
            fig1 = px.line(
                filtered_df, x='timestamp', y='compound_score',
                markers=True,
                color_discrete_sequence=['#1D9E75']
            )

        fig1.update_layout(
            xaxis_title="Date",
            yaxis_title="Mood score (-1 = very negative, +1 = very positive)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[-1, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        fig1.add_hline(
            y=0, line_dash="dash",
            line_color="gray", opacity=0.5
        )
        st.plotly_chart(fig1, use_container_width=True)

        # ── Chart 2: Emotion breakdown ───────────
        st.subheader("Emotion breakdown")
        emotion_counts = (
            filtered_df['dominant_emotion']
            .value_counts()
            .reset_index()
        )
        emotion_counts.columns = ['emotion', 'count']

        fig2 = px.bar(
            emotion_counts, x='emotion', y='count',
            color='emotion',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Chart 3: Sentiment distribution ─────
        st.subheader("Sentiment distribution")
        sentiment_counts = (
            filtered_df['sentiment']
            .value_counts()
            .reset_index()
        )
        sentiment_counts.columns = ['sentiment', 'count']

        fig3 = px.pie(
            sentiment_counts,
            names='sentiment', values='count',
            color_discrete_sequence=['#1D9E75', '#D85A30', '#888780']
        )
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)

        # ── Chart 4: Mood heatmap by day ─────────
        if len(filtered_df) >= 7:
            st.subheader("Mood by day of week")
            filtered_df = filtered_df.copy()
            filtered_df['day_of_week'] = filtered_df['timestamp'].dt.day_name()

            day_order = ["Monday","Tuesday","Wednesday",
                         "Thursday","Friday","Saturday","Sunday"]
            day_mood  = (
                filtered_df.groupby('day_of_week')['compound_score']
                .mean()
                .reindex(day_order)
                .dropna()
                .reset_index()
            )
            day_mood.columns = ['day', 'avg_mood']

            fig4 = px.bar(
                day_mood, x='day', y='avg_mood',
                color='avg_mood',
                color_continuous_scale=['#D85A30', '#888780', '#1D9E75'],
                range_color=[-1, 1]
            )
            fig4.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Day of week",
                yaxis_title="Average mood score",
                coloraxis_showscale=False
            )
            fig4.add_hline(
                y=0, line_dash="dash",
                line_color="gray", opacity=0.5
            )
            st.plotly_chart(fig4, use_container_width=True)

        # ── Recent journal entries ───────────────
        st.subheader("Recent journal entries")
        recent = filtered_df.tail(5).iloc[::-1]  # newest first
        for _, row in recent.iterrows():
            sentiment_emoji = {
                "positive": "😊",
                "negative": "😔",
                "neutral":  "😐"
            }.get(row['sentiment'], "😐")
            with st.expander(
                f"{row['timestamp'].strftime('%b %d, %Y %I:%M %p')} "
                f"— {sentiment_emoji} {row['sentiment']} "
                f"({row['dominant_emotion']})"
            ):
                st.write(row['text'])
                st.markdown(
                    f"**Mood score:** `{row['compound_score']:.2f}`"
                )