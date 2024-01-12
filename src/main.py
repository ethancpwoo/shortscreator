import os

import json
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyttsx3
import scipy.io.wavfile as wav
import whisper_timestamped as whisper


def create_subtitle(word, i, j): 

    print("Creating word... \n")

    image = Image.new("RGBA", (1080, 1920), (255, 0, 0, 0))
    shadowcolor = 'black'

    font = ImageFont.truetype("./fonts/Bangers/Bangers-Regular.ttf", 120)
    draw = ImageDraw.Draw(image)
    
    _, _, w, h = draw.textbbox((0, 0), word, font=font)
    x, y = (1080-w)/2, (1920-h)/2

    draw.text((x-2, y-2), word, font=font, fill=shadowcolor)
    draw.text((x+2, y-2), word, font=font, fill=shadowcolor)
    draw.text((x-2, y+2), word, font=font, fill=shadowcolor)
    draw.text((x+2, y+2), word, font=font, fill=shadowcolor)

    draw.text((x, y), word, font=font, fill='cyan') #FF6F61
    
    image.save(f'./temp/word{str(i)}{str(j)}.png', 'PNG')


def create_captions(clip, model, arr):

    print("Transcribing... \n")
    audio = whisper.load_audio('./temp/test.wav')
    result = whisper.transcribe(model, audio, language="en")

    print(json.dumps(result, indent = 2, ensure_ascii = False))

    for i, segment in enumerate(result['segments']):
        print(f"Reading {i} segment... \n")
        for j, word_package in enumerate(segment['words']):
            start, end = word_package['start'], word_package['end']
            sentence = word_package['text'].strip()
            sentence = sentence.replace(",", "")
            sentence = sentence.replace(".", "")

            create_subtitle(sentence, i, j)

            print(f"Encoding string {sentence} into video... \n")
            
            text_clip = ImageClip(f'./temp/word{str(i)}{str(j)}.png')
            text_clip = text_clip.set_position("center")

            text_clip = text_clip.set_start(f"00:00:{str(start)}")
            print(f"00:00:{str(start)}")
            
            duration = end - start

            text_clip = text_clip.set_duration(f"00:00:{str(duration)}")
            print(f"00:00:{str(end)}")

            arr.append(text_clip)

    return arr


def main(): 

    whisper_model = whisper.load_model('base')

    family_guy_clip = VideoFileClip("./videos/s4ep18.mp4")
    gta_clip = VideoFileClip("./videos/gtavid.mp4")

    gta_clip = gta_clip.without_audio()

    family_guy_clip = family_guy_clip.subclip(t_start='00:02:25.00', t_end='00:02:50.00')
    gta_clip = gta_clip.subclip(t_start='00:00:30.00', t_end='00:00:55.00')
    family_guy_clip = family_guy_clip.set_duration('00:00:25.00')
    gta_clip = gta_clip.set_duration('00:00:25.00')

    audio_clip = family_guy_clip

    family_guy_clip = family_guy_clip.resize(width=1080, height=720)
    family_guy_clip = family_guy_clip.set_position(("center", "top"))
    gta_clip = gta_clip.resize(width=3400, height=1920)
    gta_clip = gta_clip.crop(x1=1160, x2=2240)

    # text_clip = text_clip.set_start('00:00:1.00')
    # text_clip = text_clip.set_duration('00:00:1.00')

    # text2 = text2.set_start('00:00:3.00')
    # text2 = text2.set_duration('00:00:1.00')

    clip_array = [gta_clip, family_guy_clip]
    clip_array = create_captions(audio_clip, whisper_model, clip_array)
    
    final = CompositeVideoClip(clip_array)
    final = final.set_duration('00:00:25.00')
    final.write_videofile("../out/testclip.mp4", threads=64, codec='h264_nvenc', fps=24)


if __name__ == "__main__":
    main()