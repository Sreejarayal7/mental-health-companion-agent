import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "data/chroma"

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=CHROMA_PATH)
toolkit_collection = client.get_or_create_collection(
    name="coping_toolkit",
    embedding_function=embedding_fn
)

# ── 50+ curated CBT techniques ─────────────────────
COPING_TECHNIQUES = [
    # Anxiety / Fear
    {"id": "1",  "text": "4-7-8 breathing: Inhale for 4 seconds, hold for 7, exhale for 8. Repeat 4 times. Instantly calms the nervous system during anxiety or panic.", "tags": "anxiety fear nervousness panic breathing"},
    {"id": "2",  "text": "5-4-3-2-1 grounding: Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste. Pulls you back to the present moment during anxiety.", "tags": "anxiety fear grounding mindfulness"},
    {"id": "3",  "text": "Box breathing: Inhale 4 seconds, hold 4, exhale 4, hold 4. Used by Navy SEALs to stay calm under pressure. Great for exam or performance anxiety.", "tags": "anxiety fear stress breathing calm"},
    {"id": "4",  "text": "Write down your worry and ask: What is the actual probability this will happen? What is the worst realistic outcome? Can I handle it? CBT technique for catastrophic thinking.", "tags": "anxiety fear worry overthinking cbt"},
    {"id": "5",  "text": "Progressive muscle relaxation: Tense each muscle group for 5 seconds then release. Start from toes to head. Releases physical tension caused by anxiety.", "tags": "anxiety fear tension stress relaxation"},
    {"id": "6",  "text": "Worry time technique: Schedule 15 minutes daily to worry. When anxiety hits outside that time, write the worry down and save it for worry time. Reduces constant anxious rumination.", "tags": "anxiety fear worry rumination cbt"},

    # Sadness / Grief
    {"id": "7",  "text": "Behavioural activation: When sad, do one small enjoyable activity even if you don't feel like it. Wash your face, make tea, go for a 5-minute walk. Action comes before motivation in depression.", "tags": "sadness depression grief low mood activity"},
    {"id": "8",  "text": "Write a letter to yourself from your future self. Imagine you've gotten through this hard time. What would you say to yourself today? Builds hope and self-compassion.", "tags": "sadness grief hope self-compassion writing"},
    {"id": "9",  "text": "Three good things: Every night write 3 things that went okay today, no matter how small. Trains the brain to notice positive moments even during difficult periods.", "tags": "sadness depression gratitude positivity cbt"},
    {"id": "10", "text": "Allow yourself to grieve fully for 10 minutes. Set a timer. Cry, feel, process. When timer ends, do one physical action (drink water, wash face). Controlled emotional processing.", "tags": "sadness grief crying emotional processing"},
    {"id": "11", "text": "Self-compassion break: Say to yourself — This is a moment of suffering. Suffering is part of life. May I be kind to myself in this moment. Reduces self-criticism during sadness.", "tags": "sadness grief self-compassion kindness mindfulness"},
    {"id": "12", "text": "Connect with one person today. Send a text, make a call. Social connection is the strongest predictor of emotional recovery from sadness and depression.", "tags": "sadness depression isolation connection social"},

    # Anger
    {"id": "13", "text": "STOP technique: Stop, Take a breath, Observe your feelings without reacting, Proceed mindfully. Creates a pause between trigger and response during anger.", "tags": "anger annoyance frustration impulse control cbt"},
    {"id": "14", "text": "Physical anger release: Do 20 jumping jacks, run up stairs, punch a pillow. Physical movement metabolises the adrenaline surge that anger creates in the body.", "tags": "anger frustration physical exercise release"},
    {"id": "15", "text": "Write an unsent letter expressing exactly how angry you feel. Do not hold back. Do not send it. The act of expressing anger privately reduces its intensity.", "tags": "anger frustration writing expression release"},
    {"id": "16", "text": "Ask yourself: Will this matter in 5 years? In 5 months? In 5 days? Perspective-taking reduces the emotional intensity of anger at minor frustrations.", "tags": "anger annoyance perspective cbt reframing"},
    {"id": "17", "text": "Cold water technique: Splash cold water on your face or hold ice. Activates the dive reflex which physiologically slows heart rate and reduces anger intensity.", "tags": "anger frustration physiological calming immediate"},
    {"id": "18", "text": "I-statements: Replace 'You make me angry' with 'I feel frustrated when...' Reduces defensive reactions and communicates anger constructively.", "tags": "anger communication relationships assertiveness"},

    # Stress / Overwhelm
    {"id": "19", "text": "Brain dump: Write everything in your head onto paper without organising. Every task, worry, thought. Getting it out of your head reduces cognitive overwhelm immediately.", "tags": "stress overwhelm anxiety productivity journaling"},
    {"id": "20", "text": "Eat the frog: Identify the single most important task. Do just that one thing first. Completing it reduces stress and creates momentum for everything else.", "tags": "stress overwhelm productivity procrastination focus"},
    {"id": "21", "text": "Two-minute rule: If a task takes less than 2 minutes, do it now. Clears mental clutter that contributes to feeling overwhelmed.", "tags": "stress overwhelm productivity tasks organisation"},
    {"id": "22", "text": "Body scan meditation: Lie down and slowly scan from head to toe noticing sensations without judging them. 10 minutes reduces cortisol levels measurably.", "tags": "stress overwhelm relaxation mindfulness meditation"},
    {"id": "23", "text": "Time blocking: Assign specific tasks to specific time slots. Knowing exactly when you will do something stops your brain from constantly worrying about it.", "tags": "stress overwhelm time management exams studying"},
    {"id": "24", "text": "The 80/20 rule for stress: Identify the 20% of tasks causing 80% of your stress. Focus your energy there first rather than spreading thin across everything.", "tags": "stress overwhelm productivity prioritisation"},

    # Low motivation / Depression
    {"id": "25", "text": "Micro goals: Break your goal into the smallest possible step. Not 'study for exams' but 'open the textbook to page 1'. Tiny wins rebuild motivation gradually.", "tags": "depression low motivation procrastination goals"},
    {"id": "26", "text": "Opposite action: Do the opposite of what depression tells you. Depression says stay in bed — get up. Depression says isolate — text one person. Directly counters depressive urges.", "tags": "depression low motivation behaviour activation cbt"},
    {"id": "27", "text": "Sunlight exposure for 10 minutes in the morning. Natural light regulates circadian rhythm, boosts serotonin, and is clinically proven to improve mood.", "tags": "depression low mood energy sunlight biology"},
    {"id": "28", "text": "Movement snacks: 5 minutes of movement every hour. Even a short walk releases BDNF (brain-derived neurotrophic factor) which acts like fertiliser for brain cells.", "tags": "depression low motivation movement exercise brain"},
    {"id": "29", "text": "Accomplishment list: At end of day write what you DID complete, not what you didn't. Counteracts the negativity bias that makes depression feel total and permanent.", "tags": "depression low self-esteem accomplishment positivity"},

    # Self-esteem / Worthlessness
    {"id": "30", "text": "Evidence log: When you feel worthless, write 3 pieces of evidence that contradict that belief. CBT technique to challenge automatic negative thoughts with facts.", "tags": "worthlessness self-esteem shame cbt thoughts"},
    {"id": "31", "text": "Values clarification: Write your top 5 values (honesty, creativity, kindness). Then write one small action you took today that aligns with each. Reconnects you to your identity.", "tags": "worthlessness self-esteem identity purpose values"},
    {"id": "32", "text": "Strengths journaling: Write 3 personal strengths and one example of each from your life. Read when feeling worthless. Evidence-based self-esteem builder.", "tags": "worthlessness self-esteem strengths journaling positive"},
    {"id": "33", "text": "Compare yourself to who you were yesterday, not to others. Write one way you have grown in the last month. Social comparison destroys self-esteem; personal growth builds it.", "tags": "worthlessness self-esteem comparison growth progress"},

    # Loneliness / Isolation
    {"id": "34", "text": "Reach out to one person with a specific ask: 'Want to grab chai tomorrow?' Specific invitations are easier to accept and break isolation more effectively than vague 'let's meet soon'.", "tags": "loneliness isolation connection friendship social"},
    {"id": "35", "text": "Volunteer for one hour this week. Helping others creates a sense of purpose, connection, and belonging — three of the strongest antidotes to loneliness.", "tags": "loneliness isolation purpose connection community"},
    {"id": "36", "text": "Join one online community around a specific interest (coding, books, gaming). Shared interest creates natural conversation and reduces loneliness.", "tags": "loneliness isolation online community interest social"},

    # Sleep and rest
    {"id": "37", "text": "Sleep hygiene: No screens 30 minutes before bed. Keep room cool and dark. Same sleep-wake time daily. These three changes improve sleep quality more than any supplement.", "tags": "sleep fatigue rest energy mood regulation"},
    {"id": "38", "text": "4-7-8 breathing before sleep: Activates parasympathetic nervous system, slows heart rate, and signals to body it is safe to sleep. More effective than counting sheep.", "tags": "sleep anxiety insomnia rest breathing"},
    {"id": "39", "text": "Cognitive shuffle: Imagine random unconnected images as you fall asleep (a banana, a cloud, a bicycle). Disrupts anxious thought loops that prevent sleep.", "tags": "sleep anxiety insomnia rumination rest"},

    # Mindfulness / Present moment
    {"id": "40", "text": "One mindful minute: Set a timer for 60 seconds. Focus only on your breath. When mind wanders, gently return. Builds the mindfulness muscle even with one minute daily.", "tags": "mindfulness meditation stress anxiety present moment"},
    {"id": "41", "text": "Mindful eating: Eat one meal today without phone or TV. Chew slowly, notice flavours. Mindful eating reduces stress hormones and improves mood measurably.", "tags": "mindfulness stress present moment eating habits"},
    {"id": "42", "text": "Nature exposure for 20 minutes. Studies show trees, greenery, and open sky reduce cortisol by 20%. Walk in a park, sit near a window, or even look at nature photos.", "tags": "stress anxiety mindfulness nature calm mood"},

    # Exam and academic stress
    {"id": "43", "text": "Pomodoro technique: Study 25 minutes, break 5 minutes, repeat. After 4 cycles take a 20-minute break. Prevents mental fatigue and maintains focus for long study sessions.", "tags": "exam stress studying focus productivity academic"},
    {"id": "44", "text": "Active recall over passive reading: Close the book and try to recall what you just read. More effective than re-reading. Reduces exam anxiety by building real confidence.", "tags": "exam stress studying memory academic confidence"},
    {"id": "45", "text": "Pre-exam ritual: Deep breath, remind yourself you have prepared, recall 3 times you handled a hard situation before. Primes confidence and calm before high-stakes moments.", "tags": "exam stress anxiety performance confidence ritual"},
    {"id": "46", "text": "Worst-case scenario planning: Write the absolute worst outcome of failing this exam. Then write how you would handle it. Most worst cases are survivable. Reduces catastrophic thinking.", "tags": "exam stress anxiety catastrophising cbt reframing"},

    # Relationship stress
    {"id": "47", "text": "Nonviolent communication: Express feelings as: I feel [emotion] when [situation] because I need [need]. Please would you [request]. Resolves conflict without blame or escalation.", "tags": "relationships conflict anger communication stress"},
    {"id": "48", "text": "Cooling off period: Agree with the other person to pause conflict for 30 minutes before continuing. Allows cortisol to drop and rational thinking to return.", "tags": "relationships anger conflict communication"},
    {"id": "49", "text": "Gratitude for relationship: Write 3 specific things you appreciate about the person you're in conflict with. Reduces resentment and creates emotional space for resolution.", "tags": "relationships conflict resentment gratitude forgiveness"},

    # General wellbeing
    {"id": "50", "text": "Hydration check: Drink a full glass of water right now. Dehydration of even 2% measurably increases anxiety, fatigue, and negative mood. Simple and immediate.", "tags": "general mood energy fatigue hydration immediate"},
    {"id": "51", "text": "Journaling prompt: Write for 10 minutes on 'What do I need right now?' Not what others need, not what you should do — what do YOU need. Builds self-awareness and emotional clarity.", "tags": "general self-awareness journaling clarity needs"},
    {"id": "52", "text": "Digital detox for one hour: No social media, no news. Social media increases anxiety, comparison, and FOMO measurably. One hour offline significantly reduces mental noise.", "tags": "general anxiety stress digital social media detox"},
    {"id": "53", "text": "Creative expression: Draw, doodle, sing, cook, write a poem — any creative act for 15 minutes. Activates right brain, reduces analytical overthinking, and improves mood.", "tags": "general mood creativity expression art stress"},
    {"id": "54", "text": "Gratitude letter: Write a letter of thanks to someone who positively impacted your life. You don't have to send it. Gratitude letters are one of the highest-impact happiness exercises in positive psychology.", "tags": "general gratitude happiness connection positivity"},
]

def populate_toolkit():
    """Add all techniques to ChromaDB if not already there."""
    existing = toolkit_collection.count()
    if existing >= len(COPING_TECHNIQUES):
        return  # Already populated
    print(f"Populating coping toolkit ({len(COPING_TECHNIQUES)} techniques)...")
    toolkit_collection.add(
        documents=[t["text"] for t in COPING_TECHNIQUES],
        ids=[t["id"] for t in COPING_TECHNIQUES],
        metadatas=[{"tags": t["tags"]} for t in COPING_TECHNIQUES],
    )
    print("Coping toolkit ready!")

def get_coping_techniques(emotion, sentiment, user_text, n=3):
    """
    Retrieve top N most relevant coping techniques using semantic search.
    Combines emotion + sentiment + user text as the query for best results.
    """
    try:
        # Build rich query combining emotion context + user's own words
        query = f"{emotion} {sentiment} {user_text}"
        results = toolkit_collection.query(
            query_texts=[query],
            n_results=min(n, toolkit_collection.count())
        )
        if results and results['documents'][0]:
            return results['documents'][0]
        return []
    except Exception as e:
        print(f"Coping toolkit error: {e}")
        return []

def format_techniques_for_display(techniques):
    """Format techniques as numbered list for Streamlit display."""
    if not techniques:
        return []
    formatted = []
    for i, tech in enumerate(techniques, 1):
        # Extract just the technique name (before the colon)
        if ":" in tech:
            name, desc = tech.split(":", 1)
            formatted.append({"name": name.strip(), "desc": desc.strip()})
        else:
            formatted.append({"name": f"Technique {i}", "desc": tech})
    return formatted

def format_techniques_for_prompt(techniques):
    """Format techniques as text for LLM prompt injection."""
    if not techniques:
        return "No specific techniques available."
    return "\n".join([f"{i+1}. {t}" for i, t in enumerate(techniques)])

# Populate on module load
populate_toolkit()