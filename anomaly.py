import sqlite3
import pandas as pd
from datetime import datetime, timedelta

DB_PATH = "data/journal.db"

def get_recent_entries(days=14):
    """Get entries from the last N days as a DataFrame."""
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT compound_score, sentiment, timestamp "
            "FROM entries ORDER BY timestamp DESC LIMIT 50"
        )
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows,
                          columns=['compound_score','sentiment','timestamp'])
        df['timestamp']     = pd.to_datetime(df['timestamp'])
        df['compound_score'] = df['compound_score'].astype(float)
        return df
    except Exception as e:
        print(f"Anomaly DB error: {e}")
        return pd.DataFrame()

def check_sudden_drop(df, current_score, threshold=0.4):
    """
    Detect if today's mood dropped significantly vs recent average.
    threshold=0.4 means a drop of 0.4 points on the -1 to +1 scale.
    """
    if len(df) < 3:
        return False, 0
    recent_avg  = df['compound_score'].mean()
    drop_amount = recent_avg - current_score
    if drop_amount >= threshold:
        return True, round(drop_amount, 2)
    return False, 0

def check_consecutive_negative(df, streak_threshold=3):
    """
    Detect if last N entries are all negative sentiment.
    """
    if len(df) < streak_threshold:
        return False, 0
    recent = df.head(streak_threshold)
    all_negative = all(recent['sentiment'] == 'negative')
    if all_negative:
        return True, streak_threshold
    return False, 0

def check_declining_trend(df, window=5):
    """
    Detect if mood has been steadily declining over last N entries.
    Uses linear regression slope — negative slope = declining trend.
    """
    if len(df) < window:
        return False
    recent = df.head(window)['compound_score'].values[::-1]
    # Simple slope check: is each score lower than previous?
    differences = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
    declining   = sum(1 for d in differences if d < 0)
    # If 4 out of 4 differences are negative = clear decline
    if declining >= window - 1:
        return True
    return False

def detect_anomalies(current_score, current_sentiment):
    """
    Master function — runs all checks and returns anomaly report.
    Call this after saving each new entry.

    Returns:
        dict with keys:
            detected (bool) — any anomaly found
            alerts (list)   — list of alert message strings
            severity (str)  — 'high' / 'medium' / 'low' / 'none'
    """
    df     = get_recent_entries()
    alerts = []

    if df.empty:
        return {"detected": False, "alerts": [], "severity": "none"}

    # Check 1 — sudden mood drop
    dropped, drop_amount = check_sudden_drop(df, current_score)
    if dropped:
        alerts.append(
            f"sudden_drop|Your mood has dropped significantly today "
            f"(by {drop_amount:.1f} points) compared to your recent average."
        )

    # Check 2 — consecutive negative days
    consecutive, streak = check_consecutive_negative(df)
    if consecutive:
        alerts.append(
            f"consecutive_negative|You've had {streak} negative entries "
            f"in a row. That's a tough stretch."
        )

    # Check 3 — declining trend
    declining = check_declining_trend(df)
    if declining:
        alerts.append(
            "declining_trend|Your mood has been gradually declining "
            "over your last few entries."
        )

    # Determine severity
    if len(alerts) >= 2:
        severity = "high"
    elif len(alerts) == 1:
        severity = "medium"
    else:
        severity = "none"

    return {
        "detected": len(alerts) > 0,
        "alerts":   alerts,
        "severity": severity,
    }

def get_anomaly_message(anomaly_report):
    """
    Convert anomaly report into a warm, human check-in message
    to display BEFORE the normal LLM response.
    """
    if not anomaly_report["detected"]:
        return None

    severity = anomaly_report["severity"]
    alerts   = anomaly_report["alerts"]
    types    = [a.split("|")[0] for a in alerts]

    if "consecutive_negative" in types and "sudden_drop" in types:
        return (
            "💛 **I've noticed something important** — "
            "you've had several tough days in a row, and today feels "
            "even harder than usual. I want you to know that's completely "
            "valid, and I'm really glad you're here writing. "
            "Please remember — if things feel too heavy, "
            "**iCall (9152987821)** is always there for you. 💚"
        )
    elif "consecutive_negative" in types:
        return (
            "💛 **Just checking in** — I've noticed you've had a few "
            "difficult days in a row. That can be really draining. "
            "You don't have to go through this alone — "
            "writing here is a brave step. 💚"
        )
    elif "sudden_drop" in types:
        return (
            "💛 **I noticed something** — your mood seems lower than "
            "usual today. Something must have happened, or maybe "
            "it's just been building up. Either way, I'm here. "
            "Take your time — tell me what's going on. 💚"
        )
    elif "declining_trend" in types:
        return (
            "💛 **A gentle check-in** — I've been noticing your mood "
            "has been gradually dipping over the past few days. "
            "Sometimes we don't notice these patterns ourselves. "
            "How are you really feeling? 💚"
        )
    return None