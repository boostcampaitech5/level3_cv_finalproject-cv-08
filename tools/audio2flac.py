import pandas as pd
import cv2
from tqdm import tqdm
import ffmpeg
import os
import argparse

def main(args):
    audio_list = sorted(os.listdir(args.audio_path))
    
    for audio_name in audio_list:
        idx = int(audio_name.split("_")[0])
        audio_path = os.path.join(args.audio_path, audio_name)
        audio_obj = ffmpeg.input(audio_path)
        audio_save_path = args.audio_path
        os.makedirs(audio_save_path, exist_ok=True)
        audio_obj.output(os.path.join(audio_save_path, f"{audio_name}.flac"), acodec="flac", ac=1, ar="16k").run()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--audio_path", type=str, default="../data/bommelpiano/audio_bommelpiano/train/")
    parser.add_argument("--save_path", type=str, default="../data/bommelpiano/audio_bommelpiano/train/")
    
    args = parser.parse_args()
    
    main(args)
