
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn
from typing import List, Union
from pydantic import BaseModel
from PIL import Image
from dotenv import load_dotenv
import io
import time
import sys 
import os

load_dotenv()

from StableDiffusion import create_image, create_list_image
from OpenAIClient import OpenAIClient
from VideoBot import VideoBot

app=FastAPI()

client = OpenAIClient()

class ImageGenerationRequest(BaseModel):
    prompt: List[str]
    timestamp: str
    height: int = None
    width: int = None

class ScriptGenerationRequest(BaseModel):
    prompt: str
    timestamp: str

class AudioGenerationRequest(BaseModel):
    prompt: List[str]
    timestamp: str

class VideoGenerationRequest(BaseModel):
    timestamp: str


@app.get("/")
async def home(request: Request):
    return "Application is running :)"

# @app.post("/generate-single-image/")
# def generate_image(request: Request, prompt: ImageGenerationRequest):
#     img = create_image(prompt.prompt, prompt.height, prompt.width)
#     buf = io.BytesIO()
#     img.save(buf, 'jpeg', quality=100)
#     buf.seek(0)
#     img.seek(0)
#     timestamp = int(time.time())
#     return StreamingResponse(buf, media_type="image/jpeg",
#         headers={'Content-Disposition': 'inline; filename="%s.jpg"' %(timestamp,)})

@app.post("/generate-scripts/")
def generate_script(request: Request, prompt: ScriptGenerationRequest):
    text = client.generate_script(prompt.prompt, prompt.timestamp)
    scenes, voiceovers = client.seperate_script(text, prompt.timestamp)
    return {"scenes": scenes, "voiceovers": voiceovers}

@app.post("/generate-images/")
def generate_image(request: Request, prompt: ImageGenerationRequest):
    print("generate-images")
    create_list_image(prompt.prompt, prompt.timestamp)
    return {"message": "success"}

@app.post("/generate-audio/")
def generate_audio(request: Request, prompt: AudioGenerationRequest):
    client.generate_audio(prompt.prompt, prompt.timestamp)
    return {"message": "success"}

@app.post("/compose-video/")
def compose_video(request: Request, prompt: VideoGenerationRequest):
    videobot = VideoBot(prompt.timestamp)
    videobot.merge()
    print("creating subtitle...")
    videobot.gen_subtitle()
    print("creating srt file...")
    videobot.create_srt_file()
    print("adding subtitle...")
    videobot.add_subtitle()
    return {"message": "success"}


if __name__ == "__main__":
    os.makedirs(f"result", exist_ok=True)
    uvicorn.run("app:app", host="0.0.0.0", port=14024, reload=True)