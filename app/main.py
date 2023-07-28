import math
import json
import ffmpeg
import base64
from pytube import YouTube

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.param_functions import Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, HttpUrl
from typing import Any, List, Dict, Union, Optional
from model import preprocess, predict

app = FastAPI()

class Url(BaseModel):
    url: HttpUrl

class VideoInfos(BaseModel):
    video_info: dict
    video_range: tuple

@app.post("/url")
async def inference_url(input_url: Url):
    try:
        yt = YouTube(str(input_url.url))
        yt.streams.filter(file_extension="mp4", res="720p").order_by("resolution").desc().first().download(output_path="./video", filename="01.mp4")
    except:
        return HTMLResponse(status_code=404)
    
    probe = ffmpeg.probe("./video/01.mp4")
    video_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    
    return video_info

@app.post("/submit")
async def inference_url(input_video_info: VideoInfos):  
    video_fps = int(math.ceil(int(input_video_info.video_info['r_frame_rate'].split('/')[0]) / int(input_video_info.video_info['r_frame_rate'].split('/')[1])))
    frame_range = input_video_info.video_range[1] * video_fps - input_video_info.video_range[0] * video_fps
    
    frames_with5 = preprocess(input_video_info.video_info, input_video_info.video_range)
    roll, wav = predict(frames_with5, frame_range)

    """
    결과를 base64로 encode 후 streamlit에 송신    
    """
    
    encoded_roll = base64.b64encode(roll).decode('utf-8')
    encoded_wav = base64.b64encode(wav).decode('utf-8')

    data = {'roll': encoded_roll, 'wav': encoded_wav}
    result = json.dumps(data)

    return result