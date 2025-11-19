from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch

model_id = "sir-antonie/asr_model_v2"

processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id, device_map="auto")

print("✅ Model loaded successfully from Hugging Face!")




def transcribe_en_audio(audio_path, model_id):
    # Load model + processor
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    

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
