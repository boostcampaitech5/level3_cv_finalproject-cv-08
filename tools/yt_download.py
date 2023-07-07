from pytube import YouTube
import traceback
import pandas as pd
import cv2
from tqdm import tqdm
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--csv_path", type=str, default="../data/PianoYT/pianoyt.csv", help='default="../data/PianoYT/pianoyt.csv"')
    parser.add_argument("--save_path", type=str, default="../data/videos", help='default="../data/videos"')
    
    args = parser.parse_args()
    
    music_list = pd.read_csv(args.csv_path, names=["index", "link", "train/test", "crop_minY", "crop_maxY", "crop_minX", "crop_maxX"])
    
    # yt.streams로 사용 가능한 youtube 객체(동영상, 음성 등) 확인 가능
    # filter로 원하는 형태의 stream 가져오기 가능
    # video들은 25fps

    os.makedirs(os.path.join(args.save_path, "train/"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "test/"), exist_ok=True)
    for i, (is_train, music_link, music_idx) in tqdm(music_list[['train/test', 'link', 'index']].iterrows(), total=music_list.shape[0]):
        if is_train == 1:
            output_path = os.path.join(args.save_path, "train/")
        else:
            output_path = os.path.join(args.save_path, "test/")
        try:
            yt = YouTube(music_link)
            # yt.streams.filter(mime_type="video/mp4").get_highest_resolution().download(output_path=output_path, filename_prefix=f"{music_idx}_")
            yt.streams.filter(type="audio").order_by("abr").last().download(output_path=output_path, filename_prefix=f"{music_idx}_")
        except Exception as ex:
            err_msg = traceback.format_exc()
            print(err_msg)
            print(f"failed to download : {music_idx}, {music_link}")
