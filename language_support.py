from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Makes detection deterministic (same result every time)
DetectorFactory.seed = 42

# Supported languages with their details
SUPPORTED_LANGUAGES = {
    "en": {
        "name":    "English",
        "flag":    "🇬🇧",
        "native":  "English",
        "greeting": "How are you feeling today?",
    },
    "hi": {
        "name":    "Hindi",
        "flag":    "🇮🇳",
        "native":  "हिन्दी",
        "greeting": "आज आप कैसा महसूस कर रहे हैं?",
    },
    "te": {
        "name":    "Telugu",
        "flag":    "🇮🇳",
        "native":  "తెలుగు",
        "greeting": "మీరు ఈరోజు ఎలా అనుభవిస్తున్నారు?",
    },
    "ta": {
        "name":    "Tamil",
        "flag":    "🇮🇳",
        "native":  "தமிழ்",
        "greeting": "இன்று நீங்கள் எப்படி உணர்கிறீர்கள்?",
    },
    "kn": {
        "name":    "Kannada",
        "flag":    "🇮🇳",
        "native":  "ಕನ್ನಡ",
        "greeting": "ಇಂದು ನೀವು ಹೇಗೆ ಭಾವಿಸುತ್ತಿದ್ದೀರಿ?",
    },
    "ml": {
        "name":    "Malayalam",
        "flag":    "🇮🇳",
        "native":  "മലയാളം",
        "greeting": "ഇന്ന് നിങ്ങൾക്ക് എങ്ങനെ തോന്നുന്നു?",
    },
    "mr": {
        "name":    "Marathi",
        "flag":    "🇮🇳",
        "native":  "मराठी",
        "greeting": "आज तुम्हाला कसे वाटते?",
    },
    "bn": {
        "name":    "Bengali",
        "flag":    "🇮🇳",
        "native":  "বাংলা",
        "greeting": "আজ আপনি কেমন অনুভব করছেন?",
    },
}

def detect_language(text):
    """
    Detect language of input text.
    Returns language code (e.g. 'en', 'hi', 'te')
    Falls back to 'en' if detection fails.
    """
    try:
        if not text or len(text.strip()) < 3:
            return "en"
        lang = detect(text)
        # Only return if it's a supported language
        if lang in SUPPORTED_LANGUAGES:
            return lang
        return "en"
    except LangDetectException:
        return "en"

def get_language_info(lang_code):
    """Get full language info dict for a language code."""
    return SUPPORTED_LANGUAGES.get(lang_code, SUPPORTED_LANGUAGES["en"])

def get_language_instruction(lang_code):
    """
    Returns instruction string to inject into LLM prompt
    so it responds in the correct language.
    """
    if lang_code == "en":
        return ""  # No instruction needed for English
    lang_info = get_language_info(lang_code)
    return (
        f"\nIMPORTANT: The user wrote in {lang_info['name']} ({lang_info['native']}). "
        f"You MUST respond entirely in {lang_info['name']}. "
        f"Do not use English in your response."
    )

def translate_ui_text(lang_code):
    """
    Returns UI text translations for the chat interface.
    Used to make the placeholder text match user's language.
    """
    ui_text = {
        "en": {
            "placeholder": "How are you feeling today? Write anything...",
            "thinking":    "Analysing your feelings...",
            "crisis_msg":  "I'm really concerned about you. Please reach out:",
        },
        "hi": {
            "placeholder": "आज आप कैसा महसूस कर रहे हैं? कुछ भी लिखें...",
            "thinking":    "आपकी भावनाओं का विश्लेषण हो रहा है...",
            "crisis_msg":  "मुझे आपकी बहुत चिंता है। कृपया संपर्क करें:",
        },
        "te": {
            "placeholder": "మీరు ఈరోజు ఎలా అనుభవిస్తున్నారు? ఏదైనా రాయండి...",
            "thinking":    "మీ భావాలను విశ్లేషిస్తున్నాను...",
            "crisis_msg":  "నాకు మీ గురించి చాలా ఆందోళగా ఉంది. దయచేసి సంప్రదించండి:",
        },
        "ta": {
            "placeholder": "இன்று நீங்கள் எப்படி உணர்கிறீர்கள்? எதையும் எழுதுங்கள்...",
            "thinking":    "உங்கள் உணர்வுகளை பகுப்பாய்வு செய்கிறேன்...",
            "crisis_msg":  "நான் உங்களைப் பற்றி மிகவும் கவலைப்படுகிறேன். தொடர்பு கொள்ளுங்கள்:",
        },
        "kn": {
            "placeholder": "ಇಂದು ನೀವು ಹೇಗೆ ಭಾವಿಸುತ್ತಿದ್ದೀರಿ? ಏನಾದರೂ ಬರೆಯಿರಿ...",
            "thinking":    "ನಿಮ್ಮ ಭಾವನೆಗಳನ್ನು ವಿಶ್ಲೇಷಿಸುತ್ತಿದ್ದೇನೆ...",
            "crisis_msg":  "ನನಗೆ ನಿಮ್ಮ ಬಗ್ಗೆ ತುಂಬಾ ಚಿಂತೆಯಾಗಿದೆ. ದಯವಿಟ್ಟು ಸಂಪರ್ಕಿಸಿ:",
        },
    }
    return ui_text.get(lang_code, ui_text["en"])