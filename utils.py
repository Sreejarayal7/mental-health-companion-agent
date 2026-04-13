import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import text2emotion as te

lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        label = 'positive'
    elif compound <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    return {'scores': scores, 'label': label}

def get_emotions(text):
    try:
        emotions = te.get_emotion(text)
        return emotions
    except:
        return {'Happy': 0, 'Angry': 0, 'Surprise': 0, 'Sad': 0, 'Fear': 0}

def analyze(text):
    cleaned = preprocess(text)
    sentiment = get_sentiment(text)
    emotions = get_emotions(text)
    dominant = max(emotions, key=emotions.get)
    return {
        'original': text,
        'cleaned': cleaned,
        'sentiment': sentiment,
        'emotions': emotions,
        'dominant_emotion': dominant
    }