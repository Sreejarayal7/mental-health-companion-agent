CRISIS_KEYWORDS = [
    'suicide', 'kill myself', 'want to die', 'end my life',
    'hopeless', 'cant go on', "can't go on", 'no reason to live',
    'better off dead', 'self harm', 'hurt myself', 'worthless'
]

HELPLINES = """
CRISIS SUPPORT HELPLINES:
iCall (India): 9152987821
Vandrevala Foundation: 1860-2662-345
AASRA: 9820466627
International: befrienders.org
"""

def check_crisis(text):
    text_lower = text.lower()
    for keyword in CRISIS_KEYWORDS:
        if keyword in text_lower:
            return True
    return False

def get_crisis_message():
    return HELPLINES