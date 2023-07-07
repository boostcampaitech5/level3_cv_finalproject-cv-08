import os
import glob
import time
import numpy as np
from PIL import Image
from pytube import YouTube
import matplotlib.pyplot as plt

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

def torch_preprocess(input_file_list):
    transform = transforms.Compose([
        lambda x: x.resize((900,100)),
        lambda x: np.reshape(x,(100,900,1)),
        lambda x: np.transpose(x,[2,0,1]),
        lambda x: x/255.])
    
    input_img_list = []    
    for input_file in input_file_list:
        input_img = Image.open(input_file).convert('L')
        binarr = np.array(input_img)
        input_img = Image.fromarray(binarr.astype(np.uint8))
        input_img_list.append(transform(input_img))

    torch_input_img = torch.from_numpy(np.concatenate(input_img_list)).float().cuda()
    return torch_input_img

@st.cache_data
def preprocess(user_input):
    """ 
    user_input : str(youtube link) or file(video file)
    
    1. frame load (numpy) -> 여기서 Frame 범위를 뽑은 후 유저에게 추론할 범위를 입력 받는다. (state.frame_range)
    2. crop piano key 
    """
    
    with st.spinner("Data preprocessing ..."):
        pass

    preprocess_success_msg = st.success("Data preprocessing successfully!")
    
    with st.spinner("Data loading ..."):
        # img_files : 전처리가 끝난 후 나온 image 파일 list
        img_files = glob.glob(os.path.join("/opt/ml/data/images/inference/0", "*.png"))
        img_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

        # 5 frame 씩
        frame_files = []
        for i in range(len(img_files)):
            if i >= 2 and i < len(img_files)-2:
                # i=2 : 000001.png ~ 000005.png
                file_list = [img_files[i-2], img_files[i-1], img_files[i], img_files[i+1], img_files[i+2]]
            elif i < 2:
                file_list = [img_files[i], img_files[i], img_files[i], img_files[i+1], img_files[i+2]]
            else:
                file_list = [img_files[i-2], img_files[i-1], img_files[i], img_files[i], img_files[i]]            
            frame_files.append(file_list)
            
    load_success_msg = st.success("Data loaded successfully!")    
    
    time.sleep(1)
    preprocess_success_msg.empty()
    load_success_msg.empty()
    
    return frame_files


def inference(model, frame_files):    
    min_key, max_key = 15, 65
    output_range = 250
    threshold = 0.4
    fps = 25

    with st.spinner("Data Inferencing ..."):
        start, end = state.frame_range[0]*fps, state.frame_range[1]*fps
        save_est_logit, save_est_roll = [], []
        
        inference_data = frame_files[start:end]
        for idx, frame in enumerate(inference_data):
            torch_input_img = torch_preprocess(frame)
                        
            pred_logits = model(torch.unsqueeze(torch_input_img, dim=0))
            pred_label = torch.sigmoid(pred_logits) >= threshold   
                     
            numpy_pred_logit = pred_logits.cpu().detach().numpy()
            numpy_pred_label = pred_label.cpu().detach().numpy().astype(np.int_)
            
            save_est_logit.append(numpy_pred_logit)
            save_est_roll.append(numpy_pred_label)
            
            if ((idx+1) % output_range) == 0:
                roll = np.zeros((output_range, 88))
                roll[:, min_key:max_key+1] = np.asarray(save_est_roll).squeeze()
                save_est_logit, save_est_roll = [], []
                
                with st.expander(f"See frame {start+idx+1-output_range} to {start+idx+1}."):
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

    if "url" not in state: state.url = None
    if "video" not in state: state.video = None
    
    if "frame_range" not in state: state.frame_range = None
    

    st.header("Inference")    
    st.subheader("How to upload ?")
    
    state.tab_url, state.tab_video = st.tabs([":link: URL", ":film_frames: VIDEO"])
    
    model = load_model()
    
    with state.tab_url:
        state.url = st.text_input(label="URL", placeholder="📂 Input youtube url here")
        
        if state.url:
            # inference(_url)
            pass

    with state.tab_video:
        state.video = st.file_uploader(label="VIDEO", type=["mp4", "wav", "avi"])
        
        if state.video:
            frame_files = preprocess(user_input=state.video)
            
            state.frame_range = st.slider(label="Select video range (second)", min_value=0, max_value=(len(frame_files)+1)//25, step=10, value=(50, 100))
            
            _submit = st.button(label="Submit")
            if _submit:
                inference(model, frame_files)

    st.markdown(
        """
        <style>
        div[class*="stTextInput"] div input {                
            height: 80px;
            padding: 1rem;
        }
        </style>         
        """, unsafe_allow_html=True)