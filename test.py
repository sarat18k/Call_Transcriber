import streamlit as st
import requests
import os
import subprocess
import pandas as pd
import time
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
OLLAMA_MODEL = "llama3.2:1b"  # your local Ollama model name

HEADERS_ASSEMBLYAI = {
    "authorization": ASSEMBLYAI_API_KEY,
    "content-type": "application/json"
}

UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"


def upload_audio(file):
    response = requests.post(UPLOAD_URL, headers=HEADERS_ASSEMBLYAI, data=file)
    response.raise_for_status()
    return response.json()["upload_url"]


def request_transcription(upload_url):
    json = {
        "audio_url": upload_url,
        "auto_chapters": False,
        "speaker_labels": True  # Enable speaker diarization
    }
    response = requests.post(TRANSCRIPT_URL, json=json, headers=HEADERS_ASSEMBLYAI)
    response.raise_for_status()
    return response.json()["id"]


def get_transcription_result_with_speakers(transcript_id):
    polling_endpoint = f"{TRANSCRIPT_URL}/{transcript_id}"
    while True:
        response = requests.get(polling_endpoint, headers=HEADERS_ASSEMBLYAI)
        response.raise_for_status()
        data = response.json()
        if data['status'] == 'completed':
            # Build dialogue string with speaker labels
            dialogue = ""
            for utt in data.get('utterances', []):
                speaker = utt.get('speaker', 'Speaker')
                text = utt.get('text', '')
                # Map Speaker 0 and Speaker 1 to Interviewer / Interviewee
                if speaker == "Speaker 0":
                    label = "Interviewer"
                elif speaker == "Speaker 1":
                    label = "Interviewee"
                else:
                    label = speaker
                dialogue += f"{label}: {text}\n\n"
            return dialogue
        elif data['status'] == 'error':
            raise Exception(f"Transcription failed: {data['error']}")
        st.info("Transcribing... Please wait.")
        time.sleep(3)


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


def save_report_to_csv(filename, report_dict):
    try:
        df_existing = pd.read_csv(filename)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df_existing = pd.DataFrame()

    df_new = pd.DataFrame([report_dict])
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(filename, index=False)


st.title("Interview Audio Transcription & Analysis")

uploaded_file = st.file_uploader("Upload MP3 audio file", type=["mp3", "wav", "m4a"])

prompt = st.text_area("Enter analysis prompt (e.g. 'Give skill rating and summary of interview')")

if uploaded_file and prompt:
    if st.button("Submit"):
        with st.spinner("Uploading audio..."):
            upload_url = upload_audio(uploaded_file)

        st.success("Upload complete!")

        with st.spinner("Requesting transcription..."):
            transcript_id = request_transcription(upload_url)

        transcript_text = get_transcription_result_with_speakers(transcript_id)

        with st.expander("View Transcript"):
            st.text_area("Transcription Result", transcript_text, height=300)

        st.info("Analyzing transcription with Ollama model...")

        full_prompt = transcript_text + "\n\n" + prompt
        analysis_report = analyze_with_ollama(full_prompt)

        st.subheader("Analysis Report")
        st.write(analysis_report)

        save_report_to_csv("interview_reports.csv", {
            "timestamp": datetime.now().isoformat(),
            "filename": uploaded_file.name,
            "transcript": transcript_text,
            "analysis": analysis_report
        })

        st.success(f"Report saved to interview_reports.csv")
