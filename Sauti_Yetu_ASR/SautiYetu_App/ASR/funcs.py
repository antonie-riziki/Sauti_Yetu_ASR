import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import webrtcvad
import struct
import contextlib
import wave
import io
import whisperx
import re
import html
import textwrap
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
import torch
import wave
import warnings
import nltk
import torchaudio.functional as F

from transformers import pipeline

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download("punkt_tab")


warnings.filterwarnings("ignore")

def speech_activity_and_articulation(audio_path, text):
    """
    Performs Speech Activity Detection (SAD/VAD) and calculates articulation rate.
    Returns (articulation_rate_string, figure)
    """
    # === Load audio ===
    y, sr = librosa.load(audio_path, sr=16000)  # resample to 16 kHz

    # === Compute short-term energy for plotting ===
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    energy = np.array([
        sum(abs(y[i:i+frame_length]**2))
        for i in range(0, len(y), hop_length)
    ])

    # === WebRTC Voice Activity Detection (frame-based) ===
    vad = webrtcvad.Vad(2)  # aggressiveness mode (0–3)
    bytes_audio = (y * 32767).astype(np.int16).tobytes()
    frame_duration = 30  # ms
    frame_size = int(sr * frame_duration / 1000)
    speech_frames = []
    for i in range(0, len(y), frame_size):
        frame = y[i:i + frame_size]
        if len(frame) < frame_size:
            continue
        raw = struct.pack("%dh" % len(frame), *(np.int16(frame * 32767)))
        is_speech = vad.is_speech(raw, sr)
        speech_frames.append(is_speech)

    # === Derive speech/silence segments ===
    speech_ratio = np.mean(speech_frames)
    total_duration = len(y) / sr
    active_speech_duration = total_duration * speech_ratio

    # === Articulation Rate ===
    word_count = len(text.split())
    if active_speech_duration > 0:
        articulation_rate = word_count / active_speech_duration
    else:
        articulation_rate = 0

    articulation_str = f"Articulation Rate: {articulation_rate:.2f} words/sec"

    # === Plot Speech Activity ===
    fig, ax = plt.subplots(figsize=(10, 3))
    times = np.arange(len(energy)) * hop_length / sr
    ax.plot(times, energy / np.max(energy), label="Energy/Amplitude")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Energy")
    ax.set_title("Speech Activity Detection")

    # Highlight detected speech regions
    # === Align speech_mask and times to same length ===
    speech_mask = np.repeat(speech_frames, int(frame_size / hop_length))
    
    # Pad or trim to match times length
    if len(speech_mask) < len(times):
        speech_mask = np.pad(speech_mask, (0, len(times) - len(speech_mask)), constant_values=False)
    elif len(speech_mask) > len(times):
        speech_mask = speech_mask[:len(times)]
    
    # Now plot safely
    ax.fill_between(times, 0, 1, where=speech_mask, color="orange", alpha=0.3, label="Speech Detected")

    ax.legend(loc="upper right")
    plt.tight_layout()

    return articulation_str, fig




def articulation_clarity(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    S = np.abs(librosa.stft(y))
    spectral_centroid = librosa.feature.spectral_centroid(S=S)[0]
    spectral_entropy = -np.sum((S / np.sum(S, axis=0)) * np.log2(S / np.sum(S, axis=0) + 1e-10), axis=0)

    # Normalize clarity score (higher = clearer)
    clarity_score = np.mean(spectral_centroid) / (np.mean(spectral_entropy) + 1e-9)
    clarity_score = np.clip(clarity_score / 100, 0, 10)

    # Plot articulation trend
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(spectral_centroid, label='Spectral Centroid')
    ax2 = ax.twinx()
    ax2.plot(spectral_entropy, color='r', alpha=0.6, label='Spectral Entropy')
    ax.set_title(f"Articulation Clarity Trend (Score ≈ {clarity_score:.2f}/10)")
    ax.set_xlabel("Frames")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()

    return clarity_score, fig




def estimate_gender_age(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=50, fmax=500, sr=sr
    )
    f0 = f0[~np.isnan(f0)]
    if len(f0) == 0:
        return "Unknown", "Unknown"

    mean_pitch = np.mean(f0)

    # --- Heuristic thresholds ---
    if mean_pitch < 140:
        gender = "Male"
    elif mean_pitch < 250:
        gender = "Female"
    else:
        gender = "Child / High-pitched Voice"

    # Rough estimate of age group based on pitch range
    if mean_pitch < 120:
        age_group = "Adult Male (20–60)"
    elif mean_pitch < 200:
        age_group = "Adult Female (20–60)"
    elif mean_pitch < 300:
        age_group = "Teen / Young Adult"
    else:
        age_group = "Child (below 12)"

    return gender, age_group





def forced_alignment_plot(audio_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # STEP 1 — LOAD WHISPER MODEL
    model = whisperx.load_model("medium", device)

    # load audio
    audio, sr = torchaudio.load(audio_file)

    # STEP 2 — TRANSCRIBE
    result = model.transcribe(audio_file, batch_size=16)

    # STEP 3 — LOAD ALIGNMENT MODEL
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )

    # STEP 4 — ALIGN
    aligned = whisperx.align(
        result["segments"], model_a, metadata, audio_file, device
    )

    # STEP 5 — GENERATE SPECTROGRAM
    spec = whisperx.log_mel_spectrogram(audio, sr)
    spec = spec.squeeze().cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(spec, aspect="auto", origin="lower")
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")

    # OVERLAY ALIGNMENT
    for w in aligned["word_segments"]:
        if "start" in w and "end" in w:
            start, end = w["start"], w["end"]
            ax.plot([start*100, end*100], [10, 10], linewidth=3)

    # SAVE FIGURE TO BYTES
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return buf






# from funcs.transcriptions import transcribe_en_audio

# Path to Hugging Face Whisper model
model_path = "sir-antonie/asr_tensor_model"
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
emotion_analyzer = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", return_all_scores=True)
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def word_count(text):
    return len(text.split())


def sentiment_analysis(text):
    analysis_results = {}
    sentiment = sentiment_analyzer(text[:512])[0]  # Limit length for efficiency
    analysis_results["Sentiment"] = {
        "Label": sentiment["label"],
        "Score": round(sentiment["score"], 3)
    }

    return analysis_results


def emotion_recognition(text):
    analysis_results = {}
    emotions = emotion_analyzer(text[:512])[0]
    top_emotion = max(emotions, key=lambda x: x["score"])
    analysis_results["Emotion Recognition"] = {
        "Dominant Emotion": top_emotion["label"],
        "Confidence": round(top_emotion["score"], 3)
    }

    return analysis_results


def politeness_empathy_analysis(text):
    analysis_results = {}
    empathy_labels = ["polite", "neutral", "impolite", "empathetic", "cold"]
    empathy_pred = zero_shot(text, empathy_labels)
    top_label = empathy_pred["labels"][0]
    top_score = empathy_pred["scores"][0]
    analysis_results["Politeness/Empathy Level"] = {
        "Label": top_label,
        "Confidence": round(top_score, 3)
    }

    return analysis_results


def conversion_mood_analysis(text):
    analysis_results = {}
    mood_labels = ["friendly", "formal", "sarcastic", "informative", "enthusiastic", "calm", "tense"]
    mood_pred = zero_shot(text, mood_labels)
    analysis_results["Conversational Mood"] = {
        "Mood": mood_pred["labels"][0],
        "Confidence": round(mood_pred["scores"][0], 3)
    }

    return analysis_results


def keyword_analysis(text):
    analysis_results = {}
    keywords = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())  # words ≥ 5 letters
    freq = {}
    for word in keywords:
        freq[word] = freq.get(word, 0) + 1
    top_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
    analysis_results["Keyword Spotting"] = [w for w, _ in top_keywords]

    return analysis_results


def intent_analysis(text):
    analysis_results = {}
    intents = ["request", "command", "inform", "question", "greeting", "complaint", "opinion"]
    intent_pred = zero_shot(text, intents)
    analysis_results["Intent Recognition"] = {
        "Intent": intent_pred["labels"][0],
        "Confidence": round(intent_pred["scores"][0], 3)
    }

    return analysis_results


def dialogue_act_analysis(text):
    analysis_results = {}
    dialogue_acts = [
        "statement", "question", "agreement", "disagreement",
        "request", "suggestion", "acknowledgment"
    ]
    dialogue_pred = zero_shot(text, dialogue_acts)
    analysis_results["Dialogue Act Classification"] = {
        "Act": dialogue_pred["labels"][0],
        "Confidence": round(dialogue_pred["scores"][0], 3)
    }

    return analysis_results



def communication_efficiency_analysis(text):
    analysis_results = {}
    # Estimate clarity based on intent confidence vs linguistic effort
    word_count = len(text.split())
    clarity_score = 0.0
    effort_score = np.log1p(word_count)  # penalize verbosity

    if "Intent Recognition" in analysis_results:
        clarity_score = analysis_results["Intent Recognition"]["Confidence"]

    efficiency = round((clarity_score / effort_score) * 10, 2)
    analysis_results["Communication Efficiency Score"] = f"{efficiency} / 10"

    return analysis_results


def speech_activity_analysis(audio_path, text):
    analysis_results = {}
    figs = []

    try:
        articulation_str, fig = speech_activity_and_articulation(audio_path, text)
        analysis_results["Articulation Rate"] = articulation_str
        if "Figures" not in analysis_results:
            analysis_results["Figures"] = figs
        analysis_results["Figures"].append(("Speech Activity Detection", fig))
    except Exception as e:
        analysis_results["Speech Activity Error"] = str(e)

    return analysis_results, figs



def articulation_analysis(audio_path):
    analysis_results = {}
    figs = []

    clarity_score, clarity_fig = articulation_clarity(audio_path)
    analysis_results["Articulation Clarity Score"] = f"{clarity_score:.2f} / 10"
    figs.append(("Articulation Clarity", clarity_fig))

    return analysis_results, figs


def demographic_analysis(audio_path):
    analysis_results = {}

    gender, age_group = estimate_gender_age(audio_path)
    analysis_results["Demographic Estimation"] = {
        "Predicted Gender": gender,
        "Estimated Age Group": age_group
    }

    return analysis_results



def word_level_analysis(audio_path):
    analysis_results = {}
    plots = []
    
    alignment_img = forced_alignment_plot(audio_path)
    plots.append(alignment_img)
    
    analysis_results["Word-level Alignment"] = alignment_img

    return analysis_results, plots
    


def pitch_analysis(audio_path):
    analysis_results = {}
    figs = []

    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Extract pitch using YIN algorithm
    frame_time = 0.02  # 20ms per frame
    frame_length = int(frame_time * sample_rate)

    pitch = F.detect_pitch_frequency(
        waveform,
        sample_rate=sample_rate,
        frame_time=frame_time
    )

    # Convert to numpy for plotting
    pitch = pitch.squeeze().numpy()

    # Replace zeros (unvoiced) with NaN for cleaner plot
    pitch[pitch == 0] = np.nan

    times = np.arange(len(pitch)) * frame_time

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, pitch, color="purple", linewidth=1.8)
    ax.set_title("Pitch Contour & Voice Ratio (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.grid(True, linestyle="--", alpha=0.5)
    figs.append(("Pitch Contour & Voice Ratio (Hz)", fig))

    # Optionally, compute average pitch
    valid_pitches = pitch[~np.isnan(pitch)]
    avg_pitch = np.mean(valid_pitches) if len(valid_pitches) > 0 else 0
    analysis_results["Average Pitch (Hz)"] = round(float(avg_pitch), 2)

    return analysis_results, figs



def spectogram_analysis(audio_path):
    analysis_results = {}
    figs = []

    waveform, sample_rate = torchaudio.load(audio_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.specgram(
        waveform.numpy()[0],
        NFFT=1024,
        Fs=sample_rate,
        noverlap=512,
        cmap="magma"
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Frequency Over Time")
    figs.append(("Frequency Over Time", fig))

    analysis_results["Figures"] = figs

    return analysis_results, figs

    

def speech_noise_ratio_analysis(audio_path):
    analysis_results = {}
    figs = []

    y, sr = librosa.load(audio_path, sr=None)

    # Short-time energy
    frame_length = 2048
    hop_length = 512
    energy = np.array([sum(abs(y[i:i+frame_length]**2))
                        for i in range(0, len(y), hop_length)])

    # Estimate signal threshold and SNR
    threshold = np.percentile(energy, 70)
    signal_energy = energy[energy >= threshold].mean()
    noise_energy = energy[energy < threshold].mean()
    snr_db = 10 * np.log10(signal_energy / (noise_energy + 1e-9))

    # Plot using fig, ax
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(energy, label="Frame Energy")
    ax.axhline(threshold, color='r', linestyle='--', label="Signal Threshold")
    ax.set_title(f"Signal-to-Noise Ratio (SNR: {snr_db:.2f} dB)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Energy")
    ax.legend()
    ax.grid(True)
    figs.append(("Speech to Noise Ratio", fig))

    analysis_results["Figures"] = figs


    return analysis_results, figs



def discourse_analysis(text):
    analysis_results = {}
    figs = []

    try:

        # --- Split text into sentences ---
        sentences = sent_tokenize(text)
        if len(sentences) < 1:
            analysis_results["Discourse Analysis"] = "Not enough sentences for discourse analysis."
        else:
            # --- Generate sentence embeddings ---
            discourse_model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = discourse_model.encode(sentences)

            # --- Compute coherence between consecutive sentences ---
            coherence_scores = [
                cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                for i in range(len(embeddings) - 1)
            ]
            avg_coherence = float(np.mean(coherence_scores))

            # --- 4️⃣ Plot coherence trend ---
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(coherence_scores, marker='o', color='teal', linewidth=2)
            ax.set_title("Discourse Coherence Across Sentences", fontsize=12)
            ax.set_xlabel("Sentence Index")
            ax.set_ylabel("Semantic Similarity (0–1)")
            ax.grid(True, linestyle='--', alpha=0.6)
            figs.append(("Discourse Coherence", fig))

            # --- 5️⃣ Save average coherence score ---
            analysis_results["Discourse Analysis"] = {
                "Average Coherence": round(avg_coherence, 3),
                "Interpretation": (
                    "High coherence — consistent topic flow." if avg_coherence > 0.7
                    else "Moderate coherence — some topic shifts."
                    if avg_coherence > 0.4
                    else "Low coherence — frequent topic changes or disorganization."
                )
            }

    except Exception as e:
        analysis_results["Discourse Analysis Error"] = str(e)