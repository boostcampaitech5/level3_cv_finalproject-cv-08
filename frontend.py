import os
import cv2
import time
import pafy
import math
import ffmpeg
import tempfile
import validators
import numpy as np
from PIL import Image
from pytube import YouTube

import torch
import torchvision.transforms as transforms

from ultralytics import YOLO
import Video2RollNet
from Roll2Wav import MIDISynth

import streamlit as st
from streamlit import session_state as state


@st.cache_resource
def video_to_roll_load_model():
    model_path = "/opt/ml/data/models/Video2RollNet.pth"
    
    model = Video2RollNet.resnet18().to(state.device)
    model.load_state_dict(torch.load(model_path, map_location=state.device))
    model.eval()
    
    return model

@st.cache_resource
def piano_detection_load_model():
    model_path = "/opt/ml/data/models/piano_detection.pt"
    
    model = YOLO(model_path)
    
    return model


def preprocess(model, video):
    transform = transforms.Compose([
        lambda x: x.resize((900, 100)),
        lambda x: np.reshape(x,(100, 900, 1)),
        lambda x: np.transpose(x,[2, 0, 1]),
        lambda x: x/255.])
    
    with st.spinner("Data Preprocessing ..."):        
        out, _ = (
            ffmpeg
            .input("./video/01.mp4", ss=state.video_range[0], t=state.video_range[1]-state.video_range[0])
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', loglevel="quiet")
            .run(capture_stdout=True)
        )
        
        frames = (
            np
            .frombuffer(out, np.uint8)
            .reshape([-1, video['height'], video['width'], 3])
        )
        
        preprocessed_frames = []
        key_detected = False
        for frame in frames:
            # Piano Detection
            if not key_detected:
                pred = model.predict(source=frame, device=state.device, verbose=False)
                if pred[0].boxes:
                    if pred[0].boxes.conf.item() > 0.8:
                        xmin, ymin, xmax, ymax = tuple(np.array(pred[0].boxes.xyxy.detach().cpu()[0], dtype=int))
                        frame = frame[ymin:ymax, xmin:xmax]
                        key_detected = True
                else:
                    continue
            else:
                frame = frame[ymin:ymax, xmin:xmax]
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = Image.fromarray(frame.astype(np.uint8))
            preprocessed_frames.append(transform(frame))

        frames = np.concatenate(preprocessed_frames)
        
    preprocess_success_msg = st.success("Data Preprocessed Successfully!")
    
    with st.spinner("Data Loading ..."):
        # 5 frame 씩
        frames_with5 = []
        for i in range(len(frames)):
            if i >= 2 and i < len(frames)-2:
                file_list = [frames[i-2], frames[i-1], frames[i], frames[i+1], frames[i+2]]
            elif i < 2:
                file_list = [frames[i], frames[i], frames[i], frames[i+1], frames[i+2]]
            else:
                file_list = [frames[i-2], frames[i-1], frames[i], frames[i], frames[i]]
            frames_with5.append(file_list)
    
    frames_with5 = torch.Tensor(np.array(frames_with5)).float().cuda()
    load_success_msg = st.success("Data Loaded Successfully!")    
    
    time.sleep(1)
    preprocess_success_msg.empty()
    load_success_msg.empty()
    
    return frames_with5


def inference(model, frames_with5):    
    min_key, max_key = 15, 65
    threshold = 0.85

    with st.spinner("Data Inferencing ..."):
        preds_roll = []
        for idx, frame in enumerate(frames_with5):
            pred_logits = model(torch.unsqueeze(frame, dim=0))
            pred_roll = torch.sigmoid(pred_logits) >= threshold   
            numpy_pred_roll = pred_roll.cpu().detach().numpy().astype(np.int_)
            preds_roll.append(numpy_pred_roll)

        preds_roll = np.asarray(preds_roll).squeeze()
        if preds_roll.shape[0] != state.frame_range:
            temp = np.zeros((state.frame_range, max_key-min_key+1))
            temp[:preds_roll.shape[0], :] = preds_roll
            preds_roll = temp

        roll = np.zeros((state.frame_range, 88))
        roll[:, min_key:max_key+1] = preds_roll
        wav, pm = MIDISynth(roll, state.frame_range).process_roll()

        st.image(np.rot90(roll, 1), width=700)
        st.audio(wav, sample_rate=16000)
    
    inference_success_msg = st.success("Data Inferenced successfully!")
    time.sleep(1)
    inference_success_msg.empty()


# streamlit run ./inference_dashboard.py --server.port 30006 --server.fileWatcherType none
if __name__ == "__main__":
    st.set_page_config(page_title="Piano To Roll")
    
    # session.state
    if "device" not in state: state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if "tab_url" not in state: state.tab_url = None
    if "tab_video" not in state: state.tab_video = None

    if "input_url" not in state: state.input_url = None
    if "input_video" not in state: state.input_video = None

    if "video_fps" not in state: state.video_fps = None
    if "video_range" not in state: state.video_range = None
    if "frame_range" not in state: state.frame_range = None
    
    st.header("Inference")    
    st.subheader("How to upload ?")
    
    st.markdown(
        """
        <style>
        div[class*="stTextInput"] div input {                
            height: 80px;
            padding: 1rem;
        }
        </style>         
        """, unsafe_allow_html=True)
    
    state.tab_url, state.tab_video = st.tabs([":link: URL", ":film_frames: VIDEO"])
    
    video_to_roll_model = video_to_roll_load_model()
    piano_detection_model = piano_detection_load_model()
    
    # cv2
    # preprocess time : 68.92283010482788
    # inference time : 10.378642082214355
    
    # ffmpeg
    # download time : 1~3
    # preprocess time : 24.259491682052612
    # inference time : 10.184748649597168
    
    with state.tab_url:
        # https://youtu.be/_3qnL9ddHuw
        state.input_url = st.text_input(label="URL", placeholder="📂 Input youtube url here (ex. https://youtu.be/...)")
        
        if state.input_url:           
            if validators.url(state.input_url):
                try:
                    with st.spinner("Url Analyzing ..."):
                        yt = YouTube(state.input_url)
                        yt.streams.filter(file_extension="mp4").order_by("resolution").desc().first().download(output_path="./video", filename="01.mp4")
                except:
                    st.error("Please input Youtube url !")
                else:
                    probe = ffmpeg.probe("./video/01.mp4")
                    video = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)      
                    state.video_fps = int(math.ceil(int(video['r_frame_rate'].split('/')[0]) / int(video['r_frame_rate'].split('/')[1])))
                    state.video_range = st.slider(label="Select video range (second)", min_value=0, max_value=int(float(video['duration'])), step=10, value=(50, 100))
                    state.frame_range = state.video_range[1] * state.video_fps - state.video_range[0] * state.video_fps
                    
                    url_submit = st.button(label="Submit", key="url_submit")
                    if url_submit:
                        frames_with5 = preprocess(piano_detection_model, video)
                        inference(video_to_roll_model, frames_with5)
                        os.remove("./video/01.mp4")
            else:
                st.error("Please input url !")
                

    with state.tab_video:
        video = st.file_uploader(label="VIDEO", type=["mp4", "wav", "avi"])

        if video:
            with open("./video/01.mp4", "wb") as f:
                f.write(video.getbuffer())

            probe = ffmpeg.probe("./video/01.mp4")
            video = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            
            state.video_fps = int(video['r_frame_rate'].split('/')[0])
            state.video_range = st.slider(label="Select video range (second)", min_value=0, max_value=int(float(video['duration'])), step=10, value=(50, 100))
            state.frame_range = state.video_range[1] * state.video_fps - state.video_range[0] * state.video_fps
            
            video_submit = st.button(label="Submit", key="video_submit")
            if video_submit:
                frame_files = preprocess(piano_detection_model, video)
                inference(video_to_roll_model, frame_files)
                os.remove("./video/01.mp4")