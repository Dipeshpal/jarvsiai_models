from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import librosa
import os
os.system("apt-get install ffmpeg or apt-get install ffmpeg")
os.system("apt-get install libsndfile1")

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")


def predict(file):
    audio, rate = librosa.load(file, sr=16000)
    print("rate", rate)
    inputs_ = tokenizer(audio, return_tensors="pt", padding="longest")
    input_values = inputs_.input_values.to("cpu")
    attention_mask = inputs_.attention_mask.to("cpu")

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)
    print(transcription)
    return transcription


from pydantic import BaseModel, Field
from opyrator.components.types import FileContent


class AudioSeparationInput(BaseModel):
    audio_file: FileContent = Field(..., mime_type="audio/mpeg")


class Output(BaseModel):
    results: str = None


def separate_audio(input: AudioSeparationInput) -> Output:
    """Separation of a music file to vocals (singing voice) and accompaniment.
    To try it out, you can use this example audio file: [audio_example.mp3](https://github.com/deezer/spleeter/raw/master/audio_example.mp3).
    """
    with open("my_file.mp3", "wb") as binary_file:
        binary_file.write(input.audio_file.as_bytes())

    print("------------------------------------")
    transcription = predict("my_file.mp3")
    print(transcription)
    return Output(transcription)
