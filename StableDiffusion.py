from diffusers import AutoPipelineForText2Image
import torch
import time
import os
import httpcore
# setattr(httpcore, 'SyncHTTPTransport', 'AsyncHTTPProxy')
from googletrans import Translator

f = open("prompt/script.txt", "r")
image_prompt = f.read()
translator = Translator()
translator.translate(image_prompt, dest='en').text

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", 
                                                # torch_dtype=torch.float16, 
                                                variant="fp16")
pipe.to(device)


def create_list_image(script, timestamp):
    os.makedirs(f"result/{timestamp}/image", exist_ok=True)

    for i in range(len(script)):
        print(i, translator.translate(script[i], dest='en').text)
        
        image = create_image(translator.translate(script[i], dest='en').text)

        image.save(f"result/{timestamp}/image/{i+1}.jpg")
        print(f"step {i} completed")

def create_image(prompt, height=1024, width=1024):
    
    start = time.time()
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0, height=height, width=width).images[0]
    end = time.time()
    print("Image generated in:", end - start)
    image.save("test.png")
    return image