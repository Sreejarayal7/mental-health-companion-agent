import os
import hashlib
from datetime import date
from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Fallback quotes if API fails
FALLBACK_QUOTES = {
    "negative": [
        ("This too shall pass. Every storm runs out of rain.", "Unknown"),
        ("You don't have to be positive all the time. It's okay to feel sad, angry or overwhelmed.", "Lori Deschene"),
        ("Even the darkest night will end and the sun will rise.", "Victor Hugo"),
        ("You are allowed to be both a masterpiece and a work in progress.", "Sophia Bush"),
        ("Tough times never last, but tough people do.", "Robert H. Schuller"),
    ],
    "positive": [
        ("Keep going. Everything you need will come to you at the perfect time.", "Unknown"),
        ("You are enough, just as you are.", "Meghan Markle"),
        ("Believe you can and you're halfway there.", "Theodore Roosevelt"),
        ("The only way to do great work is to love what you do.", "Steve Jobs"),
        ("Small steps every day lead to big changes over time.", "Unknown"),
    ],
    "neutral": [
        ("Be gentle with yourself. You are a child of the universe.", "Max Ehrmann"),
        ("Take it one day at a time.", "Unknown"),
        ("You are braver than you believe, stronger than you seem.", "A.A. Milne"),
        ("Progress, not perfection.", "Unknown"),
        ("What you are is what you have been. What you'll be is what you do now.", "Buddha"),
    ],
}

def get_mood_context(recent_sentiment, recent_emotion):
    """Map sentiment + emotion to a mood context string for the prompt."""
    if recent_sentiment == "negative":
        emotion_map = {
            "fear":       "anxious and fearful",
            "sadness":    "sad and low",
            "anger":      "frustrated and angry",
            "neutral":    "struggling and uncertain",
        }
        return emotion_map.get(recent_emotion, "going through a difficult time")
    elif recent_sentiment == "positive":
        emotion_map = {
            "joy":      "happy and joyful",
            "surprise": "excited and surprised",
            "neutral":  "feeling good",
        }
        return emotion_map.get(recent_emotion, "feeling positive")
    else:
        return "reflecting and taking things one day at a time"

def get_daily_quote(recent_sentiment="neutral",
                    recent_emotion="neutral",
                    force_refresh=False):
    """
    Generate a mood-aware quote using Groq LLM.
    Cached by date + mood so it doesn't regenerate on every rerun.
    Returns dict: {quote, author, mood_context}
    """
    mood_context = get_mood_context(recent_sentiment, recent_emotion)

    # Cache key = today's date + mood (changes daily or when mood changes)
    cache_key = hashlib.md5(
        f"{date.today()}{recent_sentiment}{recent_emotion}".encode()
    ).hexdigest()[:8]

    # Check cache file
    cache_path = f"data/quote_cache_{cache_key}.txt"
    os.makedirs("data", exist_ok=True)

    if not force_refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                lines = f.read().strip().split("\n")
                if len(lines) >= 2:
                    return {
                        "quote":        lines[0],
                        "author":       lines[1],
                        "mood_context": mood_context,
                        "cached":       True,
                    }
        except Exception:
            pass

    # Generate fresh quote from LLM
    try:
        prompt = f"""Generate one short, powerful motivational quote 
specifically for someone who is {mood_context} right now.

Rules:
- Quote must be under 25 words
- Must feel warm, human and genuine — not generic
- Can be from a real person OR original
- Must directly relate to the emotional state: {mood_context}
- Format your response EXACTLY as two lines:
Line 1: The quote text only (no quotation marks)
Line 2: Author name only (write 'Unknown' if original)

Nothing else. No explanation. Just two lines."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            stream=False,
            temperature=0.9  # Higher creativity for quotes
        )

        raw   = response.choices[0].message.content.strip()
        lines = [l.strip() for l in raw.split("\n") if l.strip()]

        if len(lines) >= 2:
            quote  = lines[0].strip('"').strip("'")
            author = lines[1]
        else:
            quote  = lines[0].strip('"').strip("'") if lines else ""
            author = "Unknown"

        # Save to cache
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(f"{quote}\n{author}")

        return {
            "quote":        quote,
            "author":       author,
            "mood_context": mood_context,
            "cached":       False,
        }

    except Exception as e:
        print(f"Quote generation error: {e}")
        # Use fallback
        import random
        fallbacks = FALLBACK_QUOTES.get(recent_sentiment,
                                        FALLBACK_QUOTES["neutral"])
        q, a = random.choice(fallbacks)
        return {
            "quote":        q,
            "author":       a,
            "mood_context": mood_context,
            "cached":       False,
        }

def get_last_mood_from_db():
    """
    Read last entry's sentiment and emotion from SQLite
    to personalise the quote.
    """
    try:
        import sqlite3
        conn   = sqlite3.connect("data/journal.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sentiment, dominant_emotion FROM entries "
            "ORDER BY timestamp DESC LIMIT 1"
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0], row[1]  # sentiment, emotion
        return "neutral", "neutral"
    except Exception:
        return "neutral", "neutral"