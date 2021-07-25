from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import librosa
from pydantic import BaseModel, Field
from opyrator.components.types import FileContent

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")


def predict(file):
    print("Recognizing...")
    input_audio, _ = librosa.load(file, sr=16000)
    input_values = tokenizer(input_audio, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    return transcription.lower()


class AudioSeparationInput(BaseModel):
    audio_file: FileContent = Field(..., mime_type="audio/wav")


class Output(BaseModel):
    results: str = Field(...)


def separate_audio(input: AudioSeparationInput) -> Output:
    """Automatic Speech Recognition using Transformers 'facebook/wav2vec2-large-960h-lv60-self'.
    It is just an API call away"""
    with open("my_file.wav", "wb") as binary_file:
        binary_file.write(input.audio_file.as_bytes())

    transcription = predict("my_file.wav")
    return Output(results=transcription)
