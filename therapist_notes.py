from groq import Groq
from reportlab.lib.pagesizes import A4 # type: ignore
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, # type: ignore
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle # type: ignore
from reportlab.lib import colors # type: ignore
from reportlab.lib.units import mm # type: ignore
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY # type: ignore
from datetime import datetime
import os
import json
from dotenv import load_dotenv

load_dotenv(override=True)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_session_note_data(conversation_history, emotion_data):
    """
    Ask LLM to analyse the conversation and generate
    structured clinical session note data as JSON.
    """
    # Build conversation text from history
    convo_text = ""
    for msg in conversation_history:
        role = "Client" if msg["role"] == "user" else "Companion"
        convo_text += f"{role}: {msg['content']}\n\n"

    prompt = f"""
You are an experienced mental health documentation assistant.
Analyse this therapy session and generate a structured session note.

SESSION TRANSCRIPT:
{convo_text}

EMOTION DATA DETECTED:
- Dominant emotion: {emotion_data.get('dominant_emotion', 'unknown')}
- Emotion group: {emotion_data.get('emotion_group', 'unknown')}
- Top emotions: {emotion_data.get('top_emotions_str', 'unknown')}
- Sentiment: {emotion_data.get('sentiment', {}).get('label', 'unknown')}
- Mood score: {emotion_data.get('sentiment', {}).get('scores', {}).get('compound', 0):.2f}

Generate a clinical session note in this EXACT JSON format.
Return ONLY the JSON, no other text:

{{
  "session_summary": "2-3 sentence summary of what the client shared",
  "presenting_emotions": ["emotion1", "emotion2", "emotion3"],
  "key_themes": ["theme1", "theme2", "theme3"],
  "cognitive_patterns": "Brief description of any thinking patterns observed",
  "coping_strategies_suggested": ["strategy1", "strategy2"],
  "client_strengths": "Positive qualities or strengths observed in the client",
  "risk_level": "low/medium/high",
  "risk_notes": "Brief risk assessment notes",
  "follow_up_recommendations": ["recommendation1", "recommendation2"],
  "session_rating": "positive/neutral/concerning",
  "clinician_notes": "Additional observations for next session"
}}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        stream=False
    )

    raw = response.choices[0].message.content.strip()

    # Clean JSON — remove markdown fences if present
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {
            "session_summary": "Session note could not be fully parsed.",
            "presenting_emotions": ["unknown"],
            "key_themes": ["general distress"],
            "cognitive_patterns": "Not analysed",
            "coping_strategies_suggested": ["breathing exercises"],
            "client_strengths": "Willingness to seek support",
            "risk_level": "low",
            "risk_notes": "No immediate risk indicators",
            "follow_up_recommendations": ["Continue journaling"],
            "session_rating": "neutral",
            "clinician_notes": "Please review session manually.",
        }


def generate_pdf(note_data, emotion_data, output_path):
    """Generate a professional clinical PDF session note."""

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        rightMargin=18*mm, leftMargin=18*mm,
        topMargin=18*mm, bottomMargin=18*mm
    )

    # ── Styles ──
    def ps(name, **kw):
        return ParagraphStyle(name, **kw)

    TITLE  = ps('T', fontSize=20, textColor=colors.HexColor('#1a1a2e'),
                alignment=TA_CENTER, fontName='Helvetica-Bold', spaceAfter=4)
    SUB    = ps('S', fontSize=11, textColor=colors.HexColor('#1D9E75'),
                alignment=TA_CENTER, spaceAfter=16)
    SEC    = ps('SE', fontSize=11, textColor=colors.white,
                fontName='Helvetica-Bold', backColor=colors.HexColor('#1a1a2e'),
                spaceBefore=10, spaceAfter=6, leftIndent=-5, borderPad=5)
    BODY   = ps('B', fontSize=10, textColor=colors.HexColor('#333333'),
                leading=15, spaceAfter=5, alignment=TA_JUSTIFY)
    LABEL  = ps('L', fontSize=9, textColor=colors.HexColor('#1D9E75'),
                fontName='Helvetica-Bold', spaceAfter=2)
    BULLET = ps('BU', fontSize=10, textColor=colors.HexColor('#333333'),
                leading=14, spaceAfter=3, leftIndent=12)

    story = []
    now   = datetime.now()

    # ── Header ──
    story.append(Paragraph("Mental Health Companion", TITLE))
    story.append(Paragraph("AI-Generated Session Note", SUB))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor('#1D9E75')))
    story.append(Spacer(1, 10))

    # ── Session metadata ──
    risk_color = {
        "low":    colors.HexColor('#1D9E75'),
        "medium": colors.HexColor('#BA7517'),
        "high":   colors.HexColor('#D85A30'),
    }.get(note_data.get("risk_level", "low"), colors.HexColor('#1D9E75'))

    rating_color = {
        "positive":   colors.HexColor('#1D9E75'),
        "neutral":    colors.HexColor('#888780'),
        "concerning": colors.HexColor('#D85A30'),
    }.get(note_data.get("session_rating", "neutral"), colors.HexColor('#888780'))

    meta = [
        ["Date", now.strftime("%d %B %Y")],
        ["Time", now.strftime("%I:%M %p")],
        ["Session Type", "Self-guided journaling companion"],
        ["Risk Level", note_data.get("risk_level", "low").upper()],
        ["Session Rating", note_data.get("session_rating", "neutral").upper()],
        ["Mood Score", f"{emotion_data.get('sentiment', {}).get('scores', {}).get('compound', 0):.2f} (-1 to +1)"],
        ["Dominant Emotion", note_data.get("presenting_emotions", ["unknown"])[0]],
    ]
    mt = Table(meta, colWidths=[45*mm, 125*mm])
    mt.setStyle(TableStyle([
        ('FONTNAME',  (0,0),(0,-1), 'Helvetica-Bold'),
        ('FONTNAME',  (1,0),(1,-1), 'Helvetica'),
        ('FONTSIZE',  (0,0),(-1,-1), 9.5),
        ('TEXTCOLOR', (0,0),(0,-1), colors.HexColor('#1D9E75')),
        ('TEXTCOLOR', (1,0),(1,-1), colors.HexColor('#333333')),
        ('ROWBACKGROUNDS', (0,0),(-1,-1),
         [colors.HexColor('#f8f9fa'), colors.white]),
        ('BOTTOMPADDING', (0,0),(-1,-1), 6),
        ('TOPPADDING',    (0,0),(-1,-1), 6),
        ('LEFTPADDING',   (0,0),(-1,-1), 10),
        # Colour risk level cell
        ('TEXTCOLOR', (1,3),(1,3), risk_color),
        ('FONTNAME',  (1,3),(1,3), 'Helvetica-Bold'),
        # Colour session rating cell
        ('TEXTCOLOR', (1,4),(1,4), rating_color),
        ('FONTNAME',  (1,4),(1,4), 'Helvetica-Bold'),
    ]))
    story.append(mt)
    story.append(Spacer(1, 8))

    # ── Session Summary ──
    story.append(Paragraph("  SESSION SUMMARY", SEC))
    story.append(Paragraph(note_data.get("session_summary", ""), BODY))

    # ── Emotion Analysis ──
    story.append(Paragraph("  EMOTION ANALYSIS", SEC))
    emotions = note_data.get("presenting_emotions", [])
    story.append(Paragraph("<b>Presenting Emotions:</b>", LABEL))
    story.append(Paragraph(", ".join(emotions), BODY))
    story.append(Paragraph("<b>Top Emotions Detected:</b>", LABEL))
    story.append(Paragraph(
        emotion_data.get("top_emotions_str", "Not available"), BODY))
    story.append(Paragraph("<b>Cognitive Patterns Observed:</b>", LABEL))
    story.append(Paragraph(
        note_data.get("cognitive_patterns", "None noted"), BODY))

    # ── Key Themes ──
    story.append(Paragraph("  KEY THEMES IDENTIFIED", SEC))
    for theme in note_data.get("key_themes", []):
        story.append(Paragraph(f"&#8226;  {theme}", BULLET))

    # ── Client Strengths ──
    story.append(Paragraph("  CLIENT STRENGTHS", SEC))
    story.append(Paragraph(
        note_data.get("client_strengths", "Willingness to seek support"), BODY))

    # ── Coping Strategies ──
    story.append(Paragraph("  COPING STRATEGIES DISCUSSED", SEC))
    for strategy in note_data.get("coping_strategies_suggested", []):
        story.append(Paragraph(f"&#8226;  {strategy}", BULLET))

    # ── Risk Assessment ──
    story.append(Paragraph("  RISK ASSESSMENT", SEC))
    risk_level = note_data.get("risk_level", "low")
    risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk_level, "🟢")
    story.append(Paragraph(
        f"<b>Risk Level:</b> {risk_emoji} {risk_level.upper()}", LABEL))
    story.append(Paragraph(
        note_data.get("risk_notes", "No immediate risk indicators"), BODY))

    # ── Follow-up Recommendations ──
    story.append(Paragraph("  FOLLOW-UP RECOMMENDATIONS", SEC))
    for rec in note_data.get("follow_up_recommendations", []):
        story.append(Paragraph(f"&#8226;  {rec}", BULLET))

    # ── Clinician Notes ──
    story.append(Paragraph("  NOTES FOR NEXT SESSION", SEC))
    story.append(Paragraph(
        note_data.get("clinician_notes", "Continue monitoring"), BODY))

    # ── Disclaimer ──
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor('#e0e0e0')))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "⚠️  DISCLAIMER: This note was generated by an AI companion, "
        "NOT a licensed mental health professional. It is intended for "
        "personal reflection only and should not be used for clinical "
        "diagnosis or treatment decisions. If you are in crisis, "
        "please contact iCall: 9152987821.",
        ps('disc', fontSize=8, textColor=colors.HexColor('#888888'),
           leading=12, alignment=TA_CENTER)))

    doc.build(story)
    return output_path


def create_session_note(conversation_history, emotion_data):
    """
    Master function — generate note data + PDF.
    Returns path to the generated PDF.
    """
    if not conversation_history:
        return None

    # Generate structured note from LLM
    note_data = generate_session_note_data(conversation_history, emotion_data)

    # Create output filename with timestamp
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/session_note_{timestamp}.pdf"
    os.makedirs("data", exist_ok=True)

    # Generate PDF
    generate_pdf(note_data, emotion_data, output_path)

    return output_path, note_data