import pandas as pd
import cv2
from tqdm import tqdm
import ffmpeg
import os
import argparse

def main(args):
    music_list = pd.read_csv(args.csv_path, names=["index", "link", "train/test", "crop_minY", "crop_maxY", "crop_minX", "crop_maxX"])
    
    vid_list = sorted(os.listdir(args.video_path))
    
    for vid_name in vid_list:
        idx = int(vid_name.split("_")[0])
        vid_path = os.path.join(args.video_path, vid_name)
        vid_obj = ffmpeg.input(vid_path)
        vid_save_path = os.path.join(args.save_path, f"{vid_name}")
        os.makedirs(vid_save_path, exist_ok=True)
        if args.no_crop:
            vid_obj.filter("fps", fps=25).filter("scale", "900", "100").output(os.path.join(vid_save_path, "%04d.png"), **{"qmin": 1, "qmax": 1}).run()
        else:
            minY, maxY, minX, maxX = music_list[music_list['index']==idx][['crop_minY', 'crop_maxY', 'crop_minX', 'crop_maxX']].values[0]
            vid_obj.filter("fps", fps=25).filter("crop", maxX-minX, maxY-minY, minX, minY).filter("scale", "900", "100").output(os.path.join(vid_save_path, "%04d.png"), **{"qmin": 1, "qmax": 1}).run()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--csv_path", type=str, default="../data/PianoYT/pianoyt.csv")
    parser.add_argument("--video_path", type=str, default="../data/videos/train")
    parser.add_argument("--save_path", type=str, default="../data/ytdataset/images/training/")
    parser.add_argument("--no_crop", action="store_true")
    
    args = parser.parse_args()
    
    main(args)
