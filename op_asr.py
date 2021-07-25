from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
from scipy.io import wavfile
import numpy as np
from scipy import interpolate
# from pydub import AudioSegment


model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")


def predict(file):
    # audio, rate = librosa.load(file, sr=16000)
    old_samplerate, old_audio = wavfile.read(file)
    NEW_SAMPLERATE = 16000
    if old_samplerate != NEW_SAMPLERATE:
        duration = old_audio.shape[0] / old_samplerate

        time_old = np.linspace(0, duration, old_audio.shape[0])
        time_new = np.linspace(0, duration, int(old_audio.shape[0] * NEW_SAMPLERATE / old_samplerate))

        interpolator = interpolate.interp1d(time_old, old_audio.T)
        new_audio = interpolator(time_new).T

        wavfile.write("out.wav", NEW_SAMPLERATE, np.round(new_audio).astype(old_audio.dtype))

    rate, audio = wavfile.read("out.wav")
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
    audio_file: FileContent = Field(..., mime_type="audio/wav")


class Output(BaseModel):
    results: str = Field(...)


def separate_audio(input: AudioSeparationInput) -> Output:
    """Separation of a music file to vocals (singing voice) and accompaniment.
    To try it out, you can use this example audio file: [audio_example.mp3](https://github.com/deezer/spleeter/raw/master/audio_example.mp3).
    """
    with open("my_file.wav", "wb") as binary_file:
        binary_file.write(input.audio_file.as_bytes())

    # print("------------------------------------")
    # sound = AudioSegment.from_mp3("my_file.mp3")
    # sound.export("my_file.wav", format="wav")
    transcription = predict("my_file.wav")
    # transcription = "transcription"
    # print(transcription)
    return Output(results=transcription)
