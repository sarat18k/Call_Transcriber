import streamlit as st
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import io

st.set_page_config(page_title="Wav2Vec2 Audio Transcriber", layout="centered")

@st.cache_resource(show_spinner=True)
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()
    return processor, model

processor, model = load_model()

def transcribe_audio_bytes(audio_bytes):
    try:
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
    except Exception as e:
        st.error(f"Could not load audio file. Error: {e}")
        return None

    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    input_values = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

st.title("🎙️ Wav2Vec2 Audio Transcription")

uploaded_file = st.file_uploader("Upload an audio file (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])

if uploaded_file:
    st.audio(uploaded_file)  # audio playback widget

    st.info("Transcribing audio, please wait...")
    transcription = transcribe_audio_bytes(uploaded_file.read())

    if transcription:
        st.subheader("Transcription Result:")
        st.markdown(f"> {transcription}")
    else:
        st.error("Failed to transcribe the audio.")

else:
    st.write("Upload an audio file above to get started.")
