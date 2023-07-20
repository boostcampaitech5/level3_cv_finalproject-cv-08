import os
import math
import time
import torch
import ffmpeg
import validators
import numpy as np
from pytube import YouTube

from preprocess import preprocess
from inference import video_to_roll_inference, roll_to_midi_inference

from ultralytics import YOLO
from models.video_to_roll import resnet18

import streamlit as st
from streamlit import session_state as state
from music21 import converter
import glob

@st.cache_resource
def video_to_roll_load_model(device):
    model_path = "./data/model/video_to_roll_best_f1.pth"
    
    model = resnet18().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model


@st.cache_resource
def roll_to_midi_load_model(device):
    model_path = "./data/model/roll_to_midi.tar"
    
    model = torch.load(model_path, map_location=device)
    
    return model


@st.cache_resource
def piano_detection_load_model(device):
    model_path = "./data/model/piano_detection.pt"
    
    model = YOLO(model_path)
    model.to(device)
    dummy_for_warmup = np.random.rand(720, 1280, 3)
    for _ in range(10):
        model.predict(source=dummy_for_warmup, device='0', verbose=False)    
    return model


# streamlit run frontend.py --server.port 30006 --server.fileWatcherType none
if __name__ == "__main__":
    
    st.set_page_config(page_title="Piano To Roll")
    
    # session.state
    if "tab_url" not in state: state.tab_url = None
    if "tab_video" not in state: state.tab_video = None

    # if "input_url" not in state: state.input_url = None
    if "prev_url" not in state: state.prev_url = None
    if "input_video" not in state: state.input_video = None
    
    if "input_video" not in state: state.input_video = None
    if "roll" not in state: state.roll = None
    if "roll_wav" not in state: state.roll_wav = None
    if "midi_wav" not in state: state.midi_wav = None
    if "submit" not in state: state.submit = False

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
    
    tab_url, tab_video = st.tabs([":link: URL", ":film_frames: VIDEO"])
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    video_to_roll_model = video_to_roll_load_model(device=device)
    roll_to_midi_model = roll_to_midi_load_model(device=device)
    piano_detection_model = piano_detection_load_model(device=device)
        
    with tab_url:
        # https://youtu.be/_3qnL9ddHuw
        # https://www.youtube.com/watch?v=ZHCjU_rcQno
        input_url = st.text_input(label="URL", placeholder="📂 Input youtube url here (ex. https://youtu.be/...)")
        if input_url:           
            if validators.url(input_url):
                try:
                    if state.prev_url != input_url:
                        state.prev_url = input_url
                        state.submit = False
                        with st.spinner("Url Analyzing ..."):
                            yt = YouTube(input_url)
                            yt.streams.filter(file_extension="mp4", res="720p").order_by("resolution").desc().first().download(output_path="./data/inference", filename="01.mp4")
                except Exception as e:
                    print(e)
                    st.error("Please check Youtube url !")
                else:
                    probe = ffmpeg.probe("./data/inference/01.mp4")
                    video_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)      
                    
                    video_info['video_fps'] = int(math.ceil(int(video_info['r_frame_rate'].split('/')[0]) / int(video_info['r_frame_rate'].split('/')[1])))
                    video_info['video_select_range'] = st.slider(label="Select video range (second)", min_value=0, max_value=int(float(video_info['duration'])), step=10, value=(50, min(int(float(video_info['duration'])), 100)), key='url')
                    video_info['video_select_frame'] = video_info['video_select_range'][1] * video_info['video_fps'] - video_info['video_select_range'][0] * video_info['video_fps']
                    
                    url_submit = st.button(label="Submit", key="url_submit")
                    if url_submit or state.submit:
                        if url_submit:
                            state.submit = True
                            
                            with st.spinner("Data Preprocessing ..."):
                                frames_with5 = preprocess(piano_detection_model, video_info, key='url')
                            preprocess_success_msg = st.success("Data Preprocessed Successfully!")
                            
                            with st.spinner("Roll Data Inferencing ..."):
                                roll, logit, roll_wav, pm_roll = video_to_roll_inference(video_to_roll_model, video_info, frames_with5)
                            roll_inference_success_msg = st.success("Data Inferenced successfully!")
                            
                            with st.spinner("Midi Data Inferencing ..."):
                                midi_wav, pm_midi = roll_to_midi_inference(roll_to_midi_model, logit)
                            midi_inference_success_msg = st.success("Data Inferenced successfully!")
                            
                            state.roll = roll
                            state.roll_wav = roll_wav
                            state.midi_wav = midi_wav
                            time.sleep(1)
                            preprocess_success_msg.empty()
                            time.sleep(0.5)
                            roll_inference_success_msg.empty()
                            time.sleep(0.5)
                            midi_inference_success_msg.empty()
                        
                            st.image(np.rot90(state.roll, 1), width=700)
                            st.audio(state.roll_wav, sample_rate=16000)
                            st.audio(state.midi_wav, sample_rate=16000)
                        
                            with st.spinner("Generating Sheet ..."):
                                output_dir = "./data/outputs"
                                os.makedirs(output_dir, exist_ok=True)
                                output_roll_midi_path = os.path.join(output_dir, "pm_roll.midi")
                                output_midi_path = os.path.join(output_dir, "pm.midi")
                                pm_roll.write(output_roll_midi_path)
                                pm_midi.write(output_midi_path)

                                roll_score = converter.parse(output_roll_midi_path)
                                midi_score = converter.parse(output_midi_path)
                                roll_pdf = os.path.join(output_dir, "roll_sheet")
                                roll_png = os.path.join(output_dir, "roll_sheet")
                                midi_pdf = os.path.join(output_dir, "midi_sheet")
                                converter.subConverters.ConverterLilypond().write(roll_score, fmt='png', fp=roll_png, subformats='png')
                                converter.subConverters.ConverterLilypond().write(roll_score, fmt='pdf', fp=roll_pdf, subformats='pdf')
                                converter.subConverters.ConverterLilypond().write(midi_score, fmt='pdf', fp=midi_pdf, subformats='pdf')
                                for file in glob.glob(output_dir + "/*[!pdf|!png]"):
                                    os.remove(file)

                            st.image("./data/outputs/roll_sheet.png")
                            state.download_click = st.download_button(
                                label="pdf download",
                                data=open("./data/outputs/roll_sheet.pdf", 'rb').read(),
                                file_name="piano_sheet.pdf",
                                mime="application/pdf"
                            )
                            state.download_click = st.download_button(
                                label="png download",
                                data=open("./data/outputs/roll_sheet.png", 'rb').read(),
                                file_name="piano_sheet.png",
                                mime="image/png"
                            )
            else:
                st.error("Please input url !")
            
    with tab_video:
        input_video = st.file_uploader(label="VIDEO", type=["mp4", "wav", "avi"])

        if input_video:
            with open("./data/inference/02.mp4", "wb") as f:
                f.write(input_video.getbuffer())

            probe = ffmpeg.probe("./data/inference/02.mp4")
            video_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)      
                    
            video_info['video_fps'] = int(math.ceil(int(video_info['r_frame_rate'].split('/')[0]) / int(video_info['r_frame_rate'].split('/')[1])))
            video_info['video_select_range'] = st.slider(label="Select video range (second)", min_value=0, max_value=int(float(video_info['duration'])), step=10, value=(50, 100), key='file')
            video_info['video_select_frame'] = video_info['video_select_range'][1] * video_info['video_fps'] - video_info['video_select_range'][0] * video_info['video_fps']
            
            video_submit = st.button(label="Submit", key="video_submit")
            if video_submit:
                with st.spinner("Data Preprocessing ..."):
                    frames_with5 = preprocess(piano_detection_model, video_info, key='file')
                preprocess_success_msg = st.success("Data Preprocessed Successfully!")
                    
                with st.spinner("Roll Data Inferencing ..."):
                    roll, logit, roll_wav, pm_roll = video_to_roll_inference(video_to_roll_model, video_info, frames_with5)
                roll_inference_success_msg = st.success("Data Inferenced successfully!")
                
                with st.spinner("Midi Data Inferencing ..."):
                    midi_wav, pm_midi = roll_to_midi_inference(roll_to_midi_model, logit)
                midi_inference_success_msg = st.success("Data Inferenced successfully!")
                
                time.sleep(1)
                preprocess_success_msg.empty()
                time.sleep(0.5)
                roll_inference_success_msg.empty()
                time.sleep(0.5)
                midi_inference_success_msg.empty()
                
                st.image(np.rot90(roll, 1), width=700)
                st.audio(roll_wav, sample_rate=16000)
                st.audio(midi_wav, sample_rate=16000)

                with st.spinner("Generating Sheet ..."):
                    output_dir = "./data/outputs"
                    os.makedirs(output_dir, exist_ok=True)
                    output_roll_midi_path = os.path.join(output_dir, "pm_roll.midi")
                    output_midi_path = os.path.join(output_dir, "pm.midi")
                    pm_roll.write(output_roll_midi_path)
                    pm_midi.write(output_midi_path)

                    roll_score = converter.parse(output_roll_midi_path)
                    midi_score = converter.parse(output_midi_path)
                    roll_pdf = os.path.join(output_dir, "roll_sheet")
                    roll_png = os.path.join(output_dir, "roll_sheet")
                    midi_pdf = os.path.join(output_dir, "midi_sheet")
                    converter.subConverters.ConverterLilypond().write(roll_score, fmt='png', fp=roll_png, subformats='png')
                    converter.subConverters.ConverterLilypond().write(roll_score, fmt='pdf', fp=roll_pdf, subformats='pdf')
                    converter.subConverters.ConverterLilypond().write(midi_score, fmt='pdf', fp=midi_pdf, subformats='pdf')
                    for file in glob.glob(output_dir + "/*[!pdf|!png]"):
                        os.remove(file)

                st.image("./data/outputs/roll_sheet.png")
                state.download_click = st.download_button(
                    label="pdf download",
                    data=open("./data/outputs/roll_sheet.pdf", 'rb').read(),
                    file_name="piano_sheet.pdf",
                    mime="application/pdf"
                )
                state.download_click = st.download_button(
                    label="png download",
                    data=open("./data/outputs/roll_sheet.png", 'rb').read(),
                    file_name="piano_sheet.png",
                    mime="image/png"
                )
                