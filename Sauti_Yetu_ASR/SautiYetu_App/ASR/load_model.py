from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torchaudio
import torch
import nltk

model_id = "sir-antonie/asr_model_v2"
nltk.download("punkt_tab")

processor = None
model = None





def get_model():
    global model, processor

    if model is None:
        print("🔥 Loading Whisper model... (one-time load)")
        model_id = "openai/whisper-base"

        processor = WhisperProcessor.from_pretrained(model_id)  
        model = WhisperForConditionalGeneration.from_pretrained(model_id, device_map="auto")
        print("✅ Model loaded successfully!")

    return model, processor






def transcribe_en_audio(audio_path, model_id):
    # Load model + processor
    model, processor = get_model()

    # processor = WhisperProcessor.from_pretrained(model_id)
    # model = WhisperForConditionalGeneration.from_pretrained(model_id)
    

    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
        

    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            # max_new_tokens=448,
            num_beams=5,
            temperature=0.7,
            repetition_penalty=1.2,
            task="transcribe",
            language="english"
        )
        

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # transcription = model.transcribe(audio_path)
    # results = transcription['text']
    return transcription
