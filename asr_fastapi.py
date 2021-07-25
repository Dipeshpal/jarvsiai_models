from fastapi import FastAPI, File, UploadFile
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import librosa

app = FastAPI()

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


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(..., mime_type="audio/wav")):
    await file.seek(0)
    a = await file.read()
    with open("my_file.wav", "wb") as f:
        f.write(a)
    ans = predict("my_file.wav")
    return {"result": ans}


if __name__ == "__main__":
    os.system("uvicorn asr_fastapi:app --reload --reload-dir data")
