from django.urls import re_path
from . import consumers #import LiveAudioConsumer

websocket_urlpatterns = [
    re_path(r"ws/live-audio/$", consumers.LiveAudioConsumer.as_asgi()),
]
