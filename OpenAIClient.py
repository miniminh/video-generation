from openai import OpenAI
from pydub import AudioSegment

from PIL import Image
import urllib.request
import requests
import json
import io
import re
import os 
from datetime import datetime


class OpenAIClient:
    def __init__(self, script_prompt_file='prompt/script.txt'):
        f = open(script_prompt_file, "r")
        self.script_prompt = f.read()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    def generate_script(self, user_input, timestamp):
        completion = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": self.script_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        print(completion.choices[0].message.content)
        os.makedirs(f"result/{timestamp}/script", exist_ok=True)
        with open(f"result/{timestamp}/script/script.txt", "w") as file:
            file.write(completion.choices[0].message.content)
        return completion.choices[0].message.content

    def seperate_script(self, text, timestamp):

        scene_pattern = re.compile(r'Scene:(.*?)(?:Voiceover:|$)', re.DOTALL)
        voiceover_pattern = re.compile(r'Voiceover:(.*?)(?:Scene:|$)', re.DOTALL)

        scenes = scene_pattern.findall(text)
        voiceovers = voiceover_pattern.findall(text)

        scenes = [scene.strip() for scene in scenes if scene.strip()]
        voiceovers = [voiceover.strip() for voiceover in voiceovers if voiceover.strip()]

        os.makedirs(f"result/{timestamp}/voiceover", exist_ok=True)
        with open(f"result/{timestamp}/voiceover/voiceover.txt", "w") as file:
            for voice in voiceovers:
                file.write(voice + "\n")
        return scenes, voiceovers

    def generate_audio(self, voiceovers, timestamp, voice='alloy', model='tts-1'):
        os.makedirs(f"result/{timestamp}/audio", exist_ok=True)
        for i in range(len(voiceovers)):
            response = self.client.audio.speech.create(
                model = model,
                voice = voice,
                input = voiceovers[i]
            )
            response.stream_to_file(f"result/{timestamp}/audio/{i+1}.mp3")
            merged_audio = AudioSegment.from_file(f"result/{timestamp}/audio/{i+1}.mp3") if i == 0 else merged_audio + AudioSegment.from_file(f"result/{timestamp}/audio/{i+1}.mp3")
        merged_audio.export(f"result/{timestamp}/audio/result.mp3", format="mp3")

# timestamp = datetime.now().strftime("%H:%M,%Y_%m_%d")
# os.makedirs(f"result/{timestamp}", exist_ok=True)
# client = OpenAIClient()
# text = client.generate_script("Làm slide và thuyết trình với nội dung giới thiệu và hướng dẫn mọi người về cách làm báo cáo tài chính.", timestamp)
# scenes, voiceovers = client.seperate_script(text, timestamp)
# client.generate_audio(voiceovers, timestamp)
