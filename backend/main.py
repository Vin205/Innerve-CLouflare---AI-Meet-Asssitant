from fastapi import FastAPI, UploadFile, File
import whisper
import openai
import os

app = FastAPI()

# Load Whisper AI Model
model = whisper.load_model("base")

# Speech-to-Text API
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    audio_path = f"temp_{file.filename}"
    with open(audio_path, "wb") as buffer:
        buffer.write(file.file.read())
    
    result = model.transcribe(audio_path)
    os.remove(audio_path)
    return {"transcription": result["text"]}

# AI Summary API (Uses OpenAI GPT)
@app.post("/summarize")
async def summarize_text(text: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Summarize this meeting: {text}"}]
    )
    return {"summary": response["choices"][0]["message"]["content"]}

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
