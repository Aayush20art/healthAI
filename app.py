import json

import streamlit as st

from voice_insights import (
    preprocess_conversation,
    build_llm_prompt,
    call_llm_real,
    call_llm_mock,
    compute_urgency,
)

st.set_page_config(page_title="Voice Insights System", page_icon="🎙️", layout="centered")

st.markdown(
    """
    <style>
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 6px;
    }
    .badge-negative { background:#fde2e1; color:#b3261e; }
    .badge-neutral  { background:#e8eaed; color:#3c4043; }
    .badge-positive { background:#e2f4e8; color:#1e7a34; }
    .badge-low      { background:#e2f4e8; color:#1e7a34; }
    .badge-medium   { background:#fff4e0; color:#a15c00; }
    .badge-high     { background:#fde2e1; color:#b3261e; }
    .badge-yes      { background:#fde2e1; color:#b3261e; }
    .badge-no       { background:#e2f4e8; color:#1e7a34; }
    .issue-chip {
        display:inline-block; background:#eef1f7; color:#333;
        padding:4px 12px; border-radius:8px; margin:3px; font-size:0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎙️ Voice Insights System")
st.caption("Turn a transcribed agent–user conversation into structured insights: summary, sentiment, risk & urgency.")

st.subheader("1. Provide the conversation")

input_mode = st.radio("Input method", ["Upload JSON file", "Paste JSON"], horizontal=True)

raw_data = None

if input_mode == "Upload JSON file":
    uploaded_file = st.file_uploader("Upload a conversation JSON file", type=["json"])
    if uploaded_file is not None:
        try:
            raw_data = json.load(uploaded_file)
        except Exception as e:
            st.error(f"Couldn't parse the uploaded file as JSON: {e}")
else:
    pasted = st.text_area(
        "Paste conversation JSON here",
        height=220,
        placeholder='{\n  "conversation": [\n    {"speaker": "agent", "text": "Hello, how can I help you today?"},\n    {"speaker": "user", "text": "I have been feeling anxious."}\n  ]\n}',
    )
    if pasted.strip():
        try:
            raw_data = json.loads(pasted)
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

show_prompt = st.checkbox("Show the exact prompt sent to the LLM", value=False)

run_clicked = st.button("Analyze conversation", type="primary", disabled=raw_data is None)

if run_clicked and raw_data is not None:
    cleaned = preprocess_conversation(raw_data)

    if not cleaned:
        st.warning("No valid conversation turns found after cleaning. Check your JSON.")
        st.stop()

    with st.expander("Cleaned conversation turns", expanded=False):
        for turn in cleaned:
            st.write(f"**{turn['speaker'].capitalize()}:** {turn['text']}")

    if show_prompt:
        with st.expander("Prompt sent to the LLM", expanded=True):
            st.code(build_llm_prompt(cleaned), language="text")

    llm_result = None
    with st.spinner("Analyzing with Mistral..."):
        try:
            llm_result = call_llm_real(cleaned)
        except Exception as e:
            st.warning(f"Mistral API call failed ({e}). Falling back to mock output.")
            llm_result = call_llm_mock(cleaned)

    urgency = compute_urgency(cleaned)

    final_output = {
        "summary": llm_result["summary"],
        "key_issues": llm_result["key_issues"],
        "sentiment": llm_result["sentiment"],
        "risk_flag": llm_result["risk_flag"],
        "urgency": urgency,
    }
 
    st.subheader("2. Insights")

    st.markdown("**Summary**")
    st.write(final_output["summary"])

    st.markdown("**Key issues**")
    chips = "".join(f'<span class="issue-chip">{issue}</span>' for issue in final_output["key_issues"])
    st.markdown(chips or "_None detected_", unsafe_allow_html=True)

    st.markdown("**Sentiment / Risk / Urgency**")
    sentiment_class = f"badge-{final_output['sentiment'].lower()}"
    risk_class = f"badge-{final_output['risk_flag']['value'].lower()}"
    urgency_class = f"badge-{urgency.lower()}"

    st.markdown(
        f"""
        <span class="badge {sentiment_class}">Sentiment: {final_output['sentiment'].capitalize()}</span>
        <span class="badge {risk_class}">Risk Flag: {final_output['risk_flag']['value']}</span>
        <span class="badge {urgency_class}">Urgency: {urgency}</span>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"**Risk reason:** {final_output['risk_flag']['reason']}")

    st.subheader("3. Raw JSON output")
    st.json(final_output)

    st.download_button(
        "Download output as JSON",
        data=json.dumps(final_output, indent=2),
        file_name="voice_insights_output.json",
        mime="application/json",
    )
