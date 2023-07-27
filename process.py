import os
import cv2
import glob
import time
import numpy as np
import soundfile
from music21 import converter

import streamlit as st
import streamlit_ext as ste
from streamlit import session_state as state
from streamlit_player import st_player

from preprocess import preprocess
from inference import video_to_roll_inference, roll_to_midi_inference

from generate_score import generate_score

def process(key):
    if key == 'url': video_path = "./data/inference/01.mp4"
    else: video_path = "./data/inference/02.mp4"
    
    video_info = dict()
    cap = cv2.VideoCapture(video_path)
    video_info['video_fps'] = int(cap.get(cv2.CAP_PROP_FPS))
    video_info['video_duration'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / video_info['video_fps'])
    
    with st.form(f"{key}_select_range_form"):
        video_info['video_select_range'] = st.slider(label="Select video range (second)", min_value=0, max_value=int(float(video_info['video_duration'])), step=10, value=(50, min(int(float(video_info['video_duration'])), 100)), key=f'{key}_silder')
        video_info['video_select_frame'] = video_info['video_select_range'][1] * video_info['video_fps'] - video_info['video_select_range'][0] * video_info['video_fps']

        col1, _, col3, col4, col5 = st.columns(5)
        with col1: 
            submit = st.form_submit_button(label="Submit")
        with col3:
            origin = st.checkbox(label="Original Video", key=f'{key}_origin_checkbox')
        with col4:
            maked = st.checkbox(label="Visualization", key=f'{key}_maked_checkbox')
        with col5: 
            sheet = st.checkbox(label="Sheet Music", value=True, disabled=True, key=f'{key}_sheet_checkbox')
    
        if submit:
            state.submit = True
            
            with st.spinner("Data Preprocess in Progress..."):
                frames_with5 = preprocess(video_info, key=key)
            preprocess_success_msg = st.success("Data has been successfully preprocessed!")
            
            with st.spinner("Roll Data Inference in Progress..."):
                roll, logit, pm_wav, pm_roll = video_to_roll_inference(video_info, frames_with5)
                np.save('./data/outputs/dump.npy', roll)
                soundfile.write("./data/outputs/sound.wav", pm_wav, 16000, format='wav')
            roll_inference_success_msg = st.success("Piano roll has been successfully inferenced!")
            
            with st.spinner("Roll Data Postprocess in Progress..."):
                midi, midi_wav, pm_midi = roll_to_midi_inference(video_info, logit)
            midi_inference_success_msg = st.success("Piano roll has been successfully postprocessed!")
            
            if maked:
                with st.spinner("Generating video ..."):
                    os.system("python game.py")
                    video_file = open("./data/outputs/video.mp4", "rb")
                    video_bytes = video_file.read()
                video_inference_success_msg = st.success("Video created successfully!")
                
                state.video_bytes = video_bytes
            else:
                state.audio_bytes = midi_wav
            
            time.sleep(0.2)
            preprocess_success_msg.empty()
            time.sleep(0.2)
            roll_inference_success_msg.empty()
            # time.sleep(0.2)
            # midi_inference_success_msg.empty()
            
            if maked:
                time.sleep(0.2)
                video_inference_success_msg.empty()

    if state.submit:
        if origin:
            with st.expander(":film_projector: Origin Video"):
                if key == 'url':
                    st_player(state.url_input, key="youtube_player")
                else:
                    st.video(state.video_input, format="video/mp4")
        
        if maked:   
            with st.expander(":film_frames: Output Video"):
                st.video(state.video_bytes, format="video/mp4")
        else:
            with st.expander(":loud_sound: Output Audio"):
                st.audio(state.audio_bytes, sample_rate=16000)

    if sheet and submit:
        state.sheet = True
        with st.spinner("Generating Music Score..."):
            output_dir = "./data/outputs"
            os.makedirs(output_dir, exist_ok=True)
            
            output_roll_midi_path = os.path.join(output_dir, "pm_roll.midi")
            output_midi_path = os.path.join(output_dir, "pm.midi")
            
            pm_roll.write(output_roll_midi_path)
            pm_midi.write(output_midi_path)
                            
            score = generate_score(output_roll_midi_path)

            roll_pdf = os.path.join(output_dir, "roll_sheet")
            roll_png = os.path.join(output_dir, "roll_sheet")
            
            converter.subConverters.ConverterLilypond().write(score, fmt='png', fp=roll_png, subformats='png')
            converter.subConverters.ConverterLilypond().write(score, fmt='pdf', fp=roll_pdf, subformats='pdf')
            
            state.sheet_file = True
            
            for file in glob.glob(output_dir + "/*"):
                if "png" not in file and "pdf" not in file:
                    os.remove(file)

    if state.sheet and state.sheet_file:
        with st.expander(":musical_note: Sheet Music"):            
            st.image("./data/outputs/roll_sheet.png")
            col1, col2, _, _, _ = st.columns(5)
            with col1:
                state.download_click = ste.download_button(
                    label="pdf download",
                    data=open("./data/outputs/roll_sheet.pdf", 'rb').read(),
                    file_name="piano_sheet.pdf",
                    mime="application/pdf"
                )
            with col2:
                state.download_click = ste.download_button(
                    label="png download",
                    data=open("./data/outputs/roll_sheet.png", 'rb').read(),
                    file_name="piano_sheet.png",
                    mime="image/png"
                )            