import os
from moviepy.editor import *
import moviepy.editor as mp
from moviepy.video.tools.subtitles import SubtitlesClip
from openai import OpenAI
import json
from moviepy.video.fx import all as vfx
import random
import cv2
import numpy as np
import math
from PIL import Image

def Zoom(clip,mode='in',position='center',speed=1.5):
    fps = 60
    duration = 2
    total_frames = int(duration*fps)
    def main(getframe,t):
        frame = getframe(t)
        h,w = frame.shape[:2]
        i = t*fps
        if mode == 'out':
            i = total_frames-i
        zoom = 1+(i*((0.1*speed)/total_frames))
        positions = {'center':[(w-(w*zoom))/2,(h-(h*zoom))/2],
                     'left':[0,(h-(h*zoom))/2],
                     'right':[(w-(w*zoom)),(h-(h*zoom))/2],
                     'top':[(w-(w*zoom))/2,0],
                     'topleft':[0,0],
                     'topright':[(w-(w*zoom)),0],
                     'bottom':[(w-(w*zoom))/2,(h-(h*zoom))],
                     'bottomleft':[0,(h-(h*zoom))],
                     'bottomright':[(w-(w*zoom)),(h-(h*zoom))]}
        tx,ty = positions[position]
        M = np.array([[zoom,0,tx], [0,zoom,ty]])
        frame = cv2.warpAffine(frame,M,(w,h))
        return frame
    return clip.fl(main)


class SlideTransition:
    def __init__(self, clip1, clip2, duration=1, direction='left'):
        self.clip1 = clip1
        self.clip2 = clip2
        self.duration = duration
        self.direction = direction

    def make_frame(self, t):
        # Calculate the position offset based on time
        offset = (self.clip1.size[0] * t / self.duration) if self.direction == 'left' else -(self.clip1.size[0] * t / self.duration)

        # Create a composite frame with the two clips shifted horizontally
        frame = VideoClip.make_composite_frame(self.clip1.get_frame(t), self.clip2.get_frame(t), (offset, 0))
        return frame

    def __getattr__(self, attr):
        # Delegate unknown attributes to the main clip
        return getattr(self.clip1, attr)


class VideoBot:
    def __init__(self, time_str) -> None:
        root = f"result/{time_str}"
        self.audio_folder = root+ "/audio"
        self.img_folder = root+ "/image"
        self.video_folder = root+ "/video"
        self.clip_folder = root+ "/clip"
        self.result = root
        os.makedirs(self.video_folder, exist_ok=True)
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = 'whisper-1'
        self.MaxWord = 5
    def merge(self):
        audio_files = os.listdir(self.audio_folder)
        img_files = os.listdir(self.img_folder)
        audio_files.sort()
        img_files.sort()
        video_clips = []

        for i, (aud, img) in enumerate(zip(audio_files, img_files)):
            aud_path = f"{self.audio_folder}/{aud}"
            img_path = f"{self.img_folder}/{img}"
            audio_clip = AudioFileClip(aud_path)

            # if i == 2:
            #     clip_path = f"{self.clip_folder}/000000.mp4"
            #     video_clip = VideoFileClip(clip_path)
            #     video_duration = video_clip.duration
            #     while video_duration < audio_clip.duration:
            #         # Lặp lại video cho đến khi độ dài của nó bằng với độ dài của audio
            #         video_clip = concatenate_videoclips([video_clip, video_clip])
            #         video_duration = video_clip.duration
            #     video_clip = video_clip.subclip((video_duration - audio_clip.duration) / 2, (video_duration + audio_clip.duration) / 2)
            #     video_clip = video_clip.resize((1920, 1080))
            #     img_clip = video_clip
            #     final_clip = CompositeVideoClip([img_clip.set_position('center')], size=(1920, 1080)).set_audio(audio_clip)
            # else:
            img_clip = ImageClip(img_path)
            img_clip = img_clip.set_duration(audio_clip.duration).set_fps(30)
            if i == 0:
                img_clip = img_clip.fadein(duration=1)
                img_clip = img_clip.set_position('center')
                img_clip = img_clip.set_position(lambda t: ('left', t * 10))
            elif i == len(img_files) - 1:
                img_clip = img_clip.fadeout(duration=1)
                img_clip = img_clip.set_position(lambda t: ('right', t * 10))
            else:
                # more than one type of moves across an image
                # img_clip = img_clip.set_position(lambda t: ('center', t * 10 + 50))
                # img_clip.fps = 30
                        # Apply different types of moves across the image
                if i % 2 == 0:
                    # Move left to right
                    img_clip = img_clip.set_position(lambda t: ('left', t * 10))
                else:
                    # Move right to left
                    img_clip = img_clip.set_position(lambda t: ('right', t * 10))

                if i % 3 == 0:
                    img_clip = Zoom(img_clip,mode='in',position='center',speed=1.2) #zoom function above
            
            final_clip = CompositeVideoClip([img_clip], size=(512, 960)).set_audio(audio_clip)
            print(final_clip)
            video_clips.append(final_clip)

        final_video = concatenate_videoclips(video_clips)
        final_video.write_videofile(f"{self.video_folder}/result.mp4", fps=24, codec="h264_nvenc")

    def gen_subtitle(self):
        audio_file = open(f"{self.audio_folder}/result.mp3", "rb")
        transcript = self.client.audio.transcriptions.create(
            file=audio_file,
            model=self.model,
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
        # store this json file
        with open(f"{self.video_folder}/transcript.json", "w",encoding='utf-8') as json_file:
            json.dump(transcript.words, json_file)

    def create_srt_file(self):
        json_file_path = f"{self.video_folder}/transcript.json"
        srt_file_path = f"{self.video_folder}/result.srt"
        voice_file_path = f"{self.result}/voiceover/voiceover.txt"
        try:
            with open(voice_file_path, "r") as file:
                voiceovers = file.readlines()
                voiceovers = [voiceover.strip() for voiceover in voiceovers if voiceover.strip()]
            voiceovers_all = " ".join(voiceovers).split()

            voiceovers_filter = [''.join(filter(lambda char: char.isalpha(), word)) for word in voiceovers_all]

            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                line = ""
                start_time = end_time = 0
                next = True
                with open(srt_file_path, 'w', encoding='utf-8') as srt_file:
                    subtitle_number = 1
                    for inx, word_data in enumerate(data):
                        if next:
                            start_time = self.format_time(word_data['start'])
                            next = False
                        end_time = self.format_time(word_data['end'])
                        #word = word_data['word']
                        try:
                            word = voiceovers_filter[inx]
                        except:
                            word = ""
                        if len(line.split()) < self.MaxWord:
                            line += word + " "
                        else:
                            srt_file.write(f"{subtitle_number}\n{start_time} --> {end_time}\n{line.strip()}\n\n")
                            line = word + " "
                            subtitle_number += 1
                            next = True
                    if line:
                        srt_file.write(f"{subtitle_number}\n{start_time} --> {end_time}\n{line.strip()}\n")

        except FileNotFoundError:
            print(f"File {json_file_path} not found.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

    @staticmethod
    def format_time(seconds):
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

    def add_subtitle(self):
        video = VideoFileClip(f"{self.video_folder}/result.mp4")
        def generator(txt):
            
            subtitle_clip = TextClip(txt, font="DVN-Poppins-ExtraBold", fontsize=30, color='white', stroke_color='black', stroke_width=2)

            return subtitle_clip
            
        subtitle = SubtitlesClip(f"{self.video_folder}/result.srt", generator)
        video1 = CompositeVideoClip([video, subtitle.set_pos(('center', 250))])
        video1.write_videofile(f"{self.video_folder}/result_with_subtitle.mp4")
