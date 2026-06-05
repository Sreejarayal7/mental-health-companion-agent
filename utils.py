import nltk
import re
import torch
from transformers import pipeline

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

lemmatizer = WordNetLemmatizer()
analyzer   = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

print("Loading GoEmotions model...")
emotion_classifier = pipeline(
    task="text-classification",
    model="monologg/bert-base-cased-goemotions-original",
    top_k=None,
    device=0 if torch.cuda.is_available() else -1
)

EMOTION_GROUPS = {
    "joy":      ["joy","amusement","excitement","gratitude","love","optimism","pride","relief"],
    "sadness":  ["sadness","disappointment","grief","remorse","embarrassment"],
    "anger":    ["anger","annoyance","disapproval","disgust"],
    "fear":     ["fear","nervousness"],
    "surprise": ["surprise","confusion","curiosity","realization"],
    "neutral":  ["neutral","caring","desire","admiration","approval"],
}

def get_emotion_group(label):
    for group, labels in EMOTION_GROUPS.items():
        if label in labels:
            return group
    return "neutral"

def preprocess(text):
    text   = text.lower()
    text   = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def get_sentiment(text):
    scores   = analyzer.polarity_scores(text)
    compound = scores['compound']
    if   compound >= 0.05:  label = 'positive'
    elif compound <= -0.05: label = 'negative'
    else:                   label = 'neutral'
    return {'scores': scores, 'label': label}

def get_emotions(text):
    try:
        if not text or len(text.strip()) < 3:
            return {"neutral": 1.0}
        results  = emotion_classifier(text[:512])[0]
        emotions = {r['label']: round(r['score'], 4) for r in results}
        return dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        print(f"GoEmotions error: {e}")
        return {"neutral": 1.0}

def get_top_emotions(text, n=3):
    emotions = get_emotions(text)
    top      = list(emotions.items())[:n]
    return ", ".join([f"{label} ({score:.0%})" for label, score in top])

def analyze(text):
    cleaned        = preprocess(text)
    sentiment      = get_sentiment(text)
    emotions       = get_emotions(text)
    dominant_label = max(emotions, key=emotions.get)
    dominant_group = get_emotion_group(dominant_label)
    top3           = get_top_emotions(text, n=3)
    return {
        'original':         text,
        'cleaned':          cleaned,
        'sentiment':        sentiment,
        'emotions':         emotions,
        'dominant_emotion': dominant_label,
        'emotion_group':    dominant_group,
        'top_emotions_str': top3,
    }