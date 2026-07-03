"""
Voice Insights System
======================
Reads a transcribed agent-user conversation (JSON) and produces structured
insights: summary, key issues, sentiment, risk flag, and urgency level.

Run:
    python voice_insights.py sample_input.json
"""

import json
import re
import sys
import requests


# ---------------------------------------------------------------------------
# TASK 1: PREPROCESSING
# ---------------------------------------------------------------------------
def preprocess_conversation(raw_data: dict) -> list:
    """
    Cleans and structures the raw conversation.
    Handles edge cases:
      - missing "conversation" key
      - empty / whitespace-only text
      - missing "speaker" or "text" fields
      - extra whitespace / repeated punctuation
    Returns a list of clean {"speaker": ..., "text": ...} dicts.
    """
    turns = raw_data.get("conversation", [])
    cleaned = []

    for turn in turns:
        speaker = turn.get("speaker", "unknown").strip().lower()
        text = turn.get("text", "")

        if text is None:
            text = ""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)          # collapse multiple spaces
        text = re.sub(r"([?.!])\1+", r"\1", text)  # collapse "??" or "!!" etc.

        if not text:
            # skip empty/blank turns instead of crashing
            continue

        cleaned.append({"speaker": speaker, "text": text})

    return cleaned


# ---------------------------------------------------------------------------
# TASK 2: NLP + PROMPT ENGINEERING (LLM extraction)
# ---------------------------------------------------------------------------

# This is the prompt that would be sent to an LLM (e.g. Claude / GPT) in a
# real deployment. It asks for strict JSON output so it's easy to parse.
LLM_PROMPT_TEMPLATE = """You are a clinical conversation analyst.
Read the following agent-user conversation transcript and analyze it.

Transcript:
{conversation_text}

Return ONLY valid JSON in exactly this format, no extra text:
{{
  "summary": "<1-2 sentence summary of the conversation>",
  "key_issues": ["<short phrase>", "<short phrase>", ...],
  "sentiment": "<positive | neutral | negative>",
  "risk_flag": {{
    "value": "<Yes | No>",
    "reason": "<short reason for the flag>"
  }}
}}

Base sentiment and risk purely on what the user says. Be concise.
"""


def build_llm_prompt(cleaned_turns: list) -> str:
    """Formats the conversation into the prompt template above."""
    convo_text = "\n".join(f"{t['speaker'].capitalize()}: {t['text']}" for t in cleaned_turns)
    return LLM_PROMPT_TEMPLATE.format(conversation_text=convo_text)


MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-large-latest"


def get_mistral_api_key() -> str:
    """
    Fetches the Mistral API key strictly from Streamlit secrets.
    Add this to .streamlit/secrets.toml:
        MISTRAL_API_KEY = "your_actual_key_here"
    """
    import streamlit as st

    key = st.secrets.get("MISTRAL_API_KEY")
    if not key:
        raise ValueError(
            "MISTRAL_API_KEY not found in Streamlit secrets. "
            "Add it to .streamlit/secrets.toml as MISTRAL_API_KEY = \"...\" "
            "(locally) or under Settings -> Secrets (on Streamlit Cloud)."
        )
    return key


def call_llm_real(cleaned_turns: list, api_key: str = None) -> dict:
    """
    REAL LLM call to the Mistral API.
    Sends the prompt built by build_llm_prompt() and asks Mistral to reply
    in strict JSON (response_format=json_object), then parses it.
    """
    api_key = api_key or get_mistral_api_key()
    prompt = build_llm_prompt(cleaned_turns)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }

    response = requests.post(MISTRAL_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    result = json.loads(content)  # Mistral returns a JSON string in `content`

    # Basic shape validation, in case the model omits a field
    result.setdefault("summary", "")
    result.setdefault("key_issues", [])
    result.setdefault("sentiment", "neutral")
    result.setdefault("risk_flag", {"value": "No", "reason": "Not provided by model"})

    return result


def call_llm_mock(cleaned_turns: list) -> dict:
    """
    FALLBACK mock LLM call — used only if the real Mistral API call fails
    (e.g. no internet, bad key, rate limit) so the pipeline never crashes.
    """
    user_text = " ".join(t["text"] for t in cleaned_turns if t["speaker"] == "user").lower()

    # --- simple heuristic "understanding" of the transcript ---
    issues = []
    if "sleep" in user_text or "insomnia" in user_text:
        issues.append("difficulty sleeping")
    if "anxious" in user_text or "anxiety" in user_text:
        issues.append("anxiety")
    if "pain" in user_text:
        issues.append("physical pain")
    if "restless" in user_text:
        issues.append("restlessness")
    if not issues:
        issues.append("general concern")

    negative_words = ["anxious", "pain", "worried", "restless", "can't", "trouble", "bad", "urgent"]
    sentiment = "negative" if any(w in user_text for w in negative_words) else "neutral"

    # Simple negation check: ignore a risk term if directly preceded by "no"/"not"/"n't"
    # e.g. "no chest pain" should NOT trigger a risk flag.
    risk_terms = ["chest pain", "suicidal", "can't breathe", "severe", "urgent"]
    risky = []
    for term in risk_terms:
        idx = user_text.find(term)
        if idx == -1:
            continue
        preceding = user_text[max(0, idx - 12):idx]
        if re.search(r"\b(no|not|n't|never)\s+$", preceding):
            continue
        risky.append(term)
    risk_flag = {
        "value": "Yes" if risky else "No",
        "reason": f"User mentioned: {', '.join(risky)}" if risky else "No high-risk keywords detected",
    }

    summary = f"User reports {', '.join(issues)}. Sentiment appears {sentiment}."

    return {
        "summary": summary,
        "key_issues": issues,
        "sentiment": sentiment,
        "risk_flag": risk_flag,
    }


# ---------------------------------------------------------------------------
# TASK 3: BASIC ML LOGIC — RULE-BASED URGENCY SCORING
# ---------------------------------------------------------------------------

# Logic: each keyword found in the user's speech adds points to a score.
# Stronger/emergency words add more points than mild ones.
# Score thresholds then map to Low / Medium / High urgency.
URGENCY_KEYWORDS = {
    "urgent": 3,
    "emergency": 3,
    "severe": 3,
    "chest pain": 3,
    "can't breathe": 3,
    "pain": 2,
    "anxious": 1,
    "anxiety": 1,
    "restless": 1,
    "trouble sleeping": 1,
    "worried": 1,
}


def compute_urgency(cleaned_turns: list) -> str:
    """
    Rule-based scoring:
      - Scan only user turns (agent speech shouldn't affect urgency).
      - Sum weights of matched keywords.
      - score >= 5  -> High
      - score 2-4   -> Medium
      - score 0-1   -> Low
    This is intentionally simple/transparent instead of a trained model,
    since the task doesn't require heavy ML.
    """
    user_text = " ".join(t["text"] for t in cleaned_turns if t["speaker"] == "user").lower()

    score = 0
    for keyword, weight in URGENCY_KEYWORDS.items():
        if keyword in user_text:
            score += weight

    if score >= 5:
        return "High"
    elif score >= 2:
        return "Medium"
    else:
        return "Low"


# ---------------------------------------------------------------------------
# TASK 4 (BONUS): AUDIO AWARENESS — pseudo-code / stub
# ---------------------------------------------------------------------------
def extract_audio_features(audio_path: str) -> dict:
    """
    PSEUDO-CODE for extracting audio features to enrich risk detection.
    No real audio is processed here (transcript-only pipeline), but this
    shows how it would work with an actual .wav/.mp3 file using librosa.

    Why this helps:
      - Pitch (fundamental frequency): a rising/shaky pitch can indicate
        distress or anxiety that plain text doesn't capture.
      - Energy/loudness: sudden spikes can indicate agitation, shouting,
        or panic; very low energy can indicate fatigue or low mood.
      - Speaking rate / pauses: rapid speech or long pauses can indicate
        nervousness or hesitation around sensitive topics.
      - Jitter/shimmer (voice quality): micro-variations linked to
        emotional or physical strain (e.g. voice cracking).

    These signals would be combined with the text-based risk_flag/urgency
    score (e.g. as extra points in compute_urgency) so the system catches
    cases where a user *sounds* distressed even if their words seem mild
    ("I'm fine" said in a shaky, high-pitched voice).
    """
    # import librosa
    # y, sr = librosa.load(audio_path)
    #
    # pitch, _ = librosa.piptrack(y=y, sr=sr)
    # avg_pitch = pitch[pitch > 0].mean()
    #
    # energy = librosa.feature.rms(y=y).mean()
    #
    # tempo, _ = librosa.beat.beat_track(y=y, sr=sr)  # proxy for speaking rate
    #
    # return {"avg_pitch": avg_pitch, "energy": energy, "tempo": tempo}

    return {
        "avg_pitch": None,
        "energy": None,
        "tempo": None,
        "note": "Stub only — no audio file processed. See docstring for real implementation plan.",
    }


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def run_pipeline(raw_data: dict, show_prompt: bool = True) -> dict:
    cleaned = preprocess_conversation(raw_data)

    if show_prompt:
        print("=" * 60)
        print("PROMPT SENT TO LLM:")
        print("=" * 60)
        print(build_llm_prompt(cleaned))
        print("=" * 60)

    try:
        llm_result = call_llm_real(cleaned)
    except Exception as e:
        print(f"[warning] Mistral API call failed ({e}); falling back to mock output.")
        llm_result = call_llm_mock(cleaned)

    urgency = compute_urgency(cleaned)

    final_output = {
        "summary": llm_result["summary"],
        "key_issues": llm_result["key_issues"],
        "sentiment": llm_result["sentiment"],
        "risk_flag": llm_result["risk_flag"],
        "urgency": urgency,
    }
    return final_output


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "sample_input.json"

    with open(input_path, "r") as f:
        data = json.load(f)

    result = run_pipeline(data)

    print("\nFINAL OUTPUT:")
    print(json.dumps(result, indent=2))