import json
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer

# Temporary placeholder
def fake_transcription():
    return "This is a sample sentence spoken by the patient"

from .views import word_count, sentiment_analysis, emotion_recognition


class LiveAudioConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.audio_buffer = []

    async def receive(self, bytes_data=None):
        if not bytes_data:
            return

        pcm = np.frombuffer(bytes_data, dtype=np.float32)
        self.audio_buffer.append(pcm)

        # Roughly 1 second of audio @16kHz
        if sum(len(b) for b in self.audio_buffer) >= 16000:
            self.audio_buffer.clear()

            text = fake_transcription()

            response = {
                "word_count": word_count(text),
                "sentiment": sentiment_analysis(text),
                "emotion": emotion_recognition(text)
            }

            await self.send(text_data=json.dumps(response))
