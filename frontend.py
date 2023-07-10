import os
import cv2
import glob
import time
import pafy
import validators
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms

import Video2RollNet
from Roll2Wav import MIDISynth

import streamlit as st
from streamlit import session_state as state


@st.cache_resource
def load_model():
    model_path = "/opt/ml/data/models/Video2RollNet.pth"
    
    device = torch.device('cuda')
    with st.spinner("Loading model ..."):
        model = Video2RollNet.resnet18().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
    success_msg = st.success("Model loaded successfully!")
    time.sleep(1)
    success_msg.empty()
    
    return model


def preprocess(video):
    """ 
    user_input : str(youtube link) or file(video file)
    
    1. frame load (numpy) -> 여기서 Frame 범위를 뽑은 후 유저에게 추론할 범위를 입력 받는다. (state.video_range)
    2. crop piano key 
    """
    transform = transforms.Compose([
        lambda x: x.resize((900, 100)),
        lambda x: np.reshape(x,(100, 900, 1)),
        lambda x: np.transpose(x,[2, 0, 1]),
        lambda x: x/255.])
    
    with st.spinner("Data preprocessing ..."):
        best = video.getbestvideo(preftype="mp4")
                    
        cap = cv2.VideoCapture(best.url)
        
        state.video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        start, end = state.video_range[0] * state.video_fps, state.video_range[1] * state.video_fps
        
        # (360, 640, 3)의 size를 가진 video의 (50 second = 1250 frame)을 처리하는 데 거리는 시간 : 27.6594 (너무 오래 걸린다)
        frames = []
        frame_count = 0

        while True:
            if frame_count < start: 
                frame_count += 1
            elif start <= frame_count < end:
                frame_count += 1
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = Image.fromarray(frame.astype(np.uint8))
                frames.append(transform(frame))

            elif (not ret) or (frame_count >= end):
                break
            
        frames = np.concatenate(frames)
    
    preprocess_success_msg = st.success("Data preprocessing successfully!")
    
    with st.spinner("Data loading ..."):
        # 5 frame 씩
        frames_with5 = np.array([])
        for i in range(len(frames)):            
            if i >= 2 and i < len(frames)-2:
                file_list = [frames[i-2], frames[i-1], frames[i], frames[i+1], frames[i+2]]
            elif i < 2:
                file_list = [frames[i], frames[i], frames[i], frames[i+1], frames[i+2]]
            else:
                file_list = [frames[i-2], frames[i-1], frames[i], frames[i], frames[i]]            
            frames_with5 = np.append(frames_with5, file_list)
    
    frames_with5 = torch.Tensor(frames_with5).float().cuda()
    
    load_success_msg = st.success("Data loaded successfully!")    
    
    time.sleep(1)
    preprocess_success_msg.empty()
    load_success_msg.empty()
    
    return frames_with5


def inference(model, frames_with5):    
    min_key, max_key = 15, 65
    output_range = state.video_range[1] * state.video_fps - state.video_range[0] * state.video_fps
    threshold = 0.4

    with st.spinner("Data Inferencing ..."):
        preds_roll = []
        for idx, frame in enumerate(frames_with5):
            pred_logits = model(torch.unsqueeze(frame, dim=0))
            pred_roll = torch.sigmoid(pred_logits) >= threshold   
            numpy_pred_roll = pred_roll.cpu().detach().numpy().astype(np.int_)

            preds_roll.append(numpy_pred_roll)
        
        roll = np.zeros((output_range, 88))
        roll[:, min_key:max_key+1] = np.asarray(preds_roll).squeeze()
        wav, pm = MIDISynth(roll, output_range).process_roll()

        st.image(np.rot90(roll, 1), width=670)
        st.audio(wav, sample_rate=16000)
    
    inference_success_msg = st.success("Inferenced successfully!")
    time.sleep(1)
    inference_success_msg.empty()


# streamlit run ./inference_dashboard.py --server.port 30006 --server.fileWatcherType none
if __name__ == "__main__":
    st.set_page_config(page_title="Piano To Roll")
    
    # session.state
    if "tab_url" not in state: state.tab_url = None
    if "tab_video" not in state: state.tab_video = None

    if "input_url" not in state: state.input_url = None
    if "input_video" not in state: state.input_video = None
    
    if "video_fps" not in state: state.video_fps = None
    if "video_range" not in state: state.video_range = None
    
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
    
    model = load_model()
    
    with state.tab_url:
        # https://youtu.be/_3qnL9ddHuw
        state.input_url = st.text_input(label="URL", placeholder="📂 Input youtube url here (ex. https://youtu.be/...)")
        
        if state.input_url:
            if validators.url(state.input_url):
                try:  video = pafy.new(state.input_url) 
                except:  st.error("Please input Youtube url !")
                else:
                    state.video_range = st.slider(label="Select video range (second)", min_value=0, max_value=video.length, step=10, value=(50, 100))
                    
                    frames_with5 = preprocess(video)

                    inference(model, frames_with5)
            else:
                st.error("Please input url !")

    # with state.tab_video:
    #     state.video = st.file_uploader(label="VIDEO", type=["mp4", "wav", "avi"])
        
    #     if state.video:
    #         frame_files = preprocess(user_input=state.video)            
    #         state.video_range = st.slider(label="Select video range (second)", min_value=0, max_value=(len(frame_files)+1)//25, step=10, value=(50, 100))
            
    #         _submit = st.button(label="Submit")
    #         if _submit:
    #             inference(model, frame_files)

