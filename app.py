import streamlit as st
import requests
import os
import subprocess
import pandas as pd
import time
import json
from dotenv import load_dotenv
from datetime import datetime

# Load API key
load_dotenv()
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
OLLAMA_MODEL = "llama3.2:1b"

HEADERS_ASSEMBLYAI = {
    "authorization": ASSEMBLYAI_API_KEY,
    "content-type": "application/json"
}

UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"

# Initialize session cache
if 'transcripts' not in st.session_state:
    st.session_state['transcripts'] = {}

# Upload to AssemblyAI
def upload_audio(file):
    st.info("Uploading file to AssemblyAI...")
    try:
        response = requests.post(UPLOAD_URL, headers=HEADERS_ASSEMBLYAI, data=file.read())
        response.raise_for_status()
        return response.json()["upload_url"]
    except requests.exceptions.RequestException as e:
        st.error(f"Upload failed: {e}")
        st.stop()

# Request transcription
def request_transcription(upload_url):
    json_data = {
        "audio_url": upload_url,
        "auto_chapters": False,
        "speaker_labels": True
    }
    try:
        response = requests.post(TRANSCRIPT_URL, json=json_data, headers=HEADERS_ASSEMBLYAI)
        response.raise_for_status()
        return response.json()["id"]
    except requests.exceptions.RequestException as e:
        st.error(f"Transcription request failed: {e}")
        st.stop()

# Poll until transcription completes
def get_transcription_result_with_speakers(transcript_id):
    polling_endpoint = f"{TRANSCRIPT_URL}/{transcript_id}"
    with st.spinner("Transcribing... Please wait."):
        while True:
            response = requests.get(polling_endpoint, headers=HEADERS_ASSEMBLYAI)
            response.raise_for_status()
            data = response.json()
            if data['status'] == 'completed':
                dialogue = ""
                for utt in data.get('utterances', []):
                    speaker = utt.get('speaker', 'Speaker')
                    label = "Interviewer" if speaker == "Speaker 0" else "Interviewee"
                    dialogue += f"{label}: {utt.get('text', '')}\n\n"
                return dialogue
            elif data['status'] == 'error':
                st.error(f"Transcription error: {data['error']}")
                st.stop()
            time.sleep(3)

# Run prompt against local LLM using Ollama
def analyze_with_ollama(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL, prompt],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error calling Ollama: {e.stderr}"

# Save analysis report to CSV
def save_report_to_csv(filename, report_dict):
    try:
        df_existing = pd.read_csv(filename)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df_existing = pd.DataFrame()

    df_new = pd.DataFrame([report_dict])
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(filename, index=False)

# Streamlit App Interface
st.title("🎤 Interview Audio Transcription & Analysis")

if not ASSEMBLYAI_API_KEY:
    st.error("AssemblyAI API key is missing. Please check your .env file.")
    st.stop()

uploaded_file = st.file_uploader("Upload MP3/WAV/M4A audio file", type=["mp3", "wav", "m4a"])
analysis_type = st.selectbox("Select analysis type", [
    "Skill Summary", "Behavioral Analysis", "Technical Depth", "Extract Q&A", "Custom Prompt"
])

custom_prompt = ""
if analysis_type == "Custom Prompt":
    custom_prompt = st.text_area("Enter your custom prompt:")

if uploaded_file:
    st.audio(uploaded_file)

if uploaded_file and st.button("Submit"):
    # Transcription (cached)
    if uploaded_file.name in st.session_state['transcripts']:
        transcript_text = st.session_state['transcripts'][uploaded_file.name]
        st.info("Loaded transcript from session cache.")
    else:
        with st.spinner("Uploading and transcribing audio..."):
            upload_url = upload_audio(uploaded_file)
            transcript_id = request_transcription(upload_url)
            transcript_text = get_transcription_result_with_speakers(transcript_id)
            st.session_state['transcripts'][uploaded_file.name] = transcript_text

    with st.expander("📄 View Transcript"):
        st.text_area("Transcript", transcript_text, height=300)

    # Prompt templates
    prompt_map = {
    "Skill Summary": (
        "Evaluate the interviewee's performance based on the following criteria:\n\n"
        "1. Communication skills\n"
        "2. Domain knowledge\n"
        "3. Confidence and clarity\n"
        "4. Soft skills (teamwork, leadership, etc.)\n\n"
        "Provide:\n"
        "- A short summary (3-5 sentences)\n"
        "- Individual scores for each category (out of 10)\n"
        "- An overall score (out of 10)\n\n"
        "Format:\n"
        "Communication: X/10\n"
        "Domain Knowledge: X/10\n"
        "Soft Skills: X/10\n"
        "Overall Score: X/10\n"
        "Summary: ..."
    ),
    "Behavioral Analysis": (
        "Based on the transcript, perform a behavioral analysis of the interviewee.\n\n"
        "Focus on:\n"
        "- Confidence\n"
        "- Leadership\n"
        "- Adaptability\n"
        "- Problem-solving\n"
        "- Communication under pressure\n\n"
        "Highlight:\n"
        "- Behavioral strengths\n"
        "- Areas of concern or improvement\n"
        "- Specific moments in the transcript that demonstrate these traits"
    ),
    "Technical Depth": (
        "Evaluate the technical depth demonstrated by the interviewee in the following areas:\n\n"
        "1. Accuracy and clarity of technical explanations\n"
        "2. Ability to solve problems or describe problem-solving strategies\n"
        "3. Understanding of core technical concepts\n"
        "4. Use of examples or real-world applications\n\n"
        "Provide:\n"
        "- A brief assessment (3-5 sentences)\n"
        "- Strengths and weaknesses\n"
        "- A technical rating out of 10"
    ),
"Extract Q&A": (
        "You are given the full interview transcript with speaker turns labeled as Interviewer and Interviewee.\n\n"
        "Extract every question asked by the Interviewer that relates to the job's technical aspects and provide the corresponding answer from the Interviewee.\n\n"
        "Ignore any personal, HR, or non-technical questions and answers.\n"
        "Only return the technical Q&A pairs that appear explicitly in the transcript.\n\n"
        "Format your output exactly like this:\n"
        "Q: [Question]\n"
        "A: [Answer]\n\n"
        "Do not include anything else—no summaries, comments, or extra text.\n"
        "Make sure to cover all relevant technical questions from the transcript."
)




}


    # Prompt selection
    base_prompt = custom_prompt if analysis_type == "Custom Prompt" else prompt_map[analysis_type]

    formatted_prompt = f"""
You are a helpful and precise AI interview assistant.

### Transcript
{transcript_text}

### Task
{base_prompt}
"""

    # Always analyze freshly using Ollama
    st.info("Running analysis with Ollama model...")
    analysis = analyze_with_ollama(formatted_prompt)

    st.subheader("🧠 Analysis Report")
    st.write(analysis)

    # Save report
    timestamp = datetime.now().isoformat()
    timestamp_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dict = {
        "timestamp": timestamp,
        "filename": uploaded_file.name,
        "analysis_type": analysis_type,
        "transcript": transcript_text,
        "analysis": analysis,
        "prompt_used": base_prompt
    }
    save_report_to_csv("interview_reports.csv", report_dict)

    # Download report
    download_filename = f"interview_analysis_{timestamp_filename}.txt"
    st.download_button("Download Report", analysis, file_name=download_filename)
    st.success("✅ Report saved and ready to download.")
