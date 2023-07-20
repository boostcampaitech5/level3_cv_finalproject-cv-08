import os
import glob
import math
import time
import numpy as np
import ffmpeg
import streamlit as st
from music21 import converter
from streamlit import session_state as state

from preprocess import preprocess
from inference import video_to_roll_inference, roll_to_midi_inference


def process(key):
    if key == 'url': video_path = "./data/inference/01.mp4"
    else: video_path = "./data/inference/02.mp4"
    
    probe = ffmpeg.probe(video_path)
    video_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)      
    
    video_info['video_fps'] = int(math.ceil(int(video_info['r_frame_rate'].split('/')[0]) / int(video_info['r_frame_rate'].split('/')[1])))
    video_info['video_select_range'] = st.slider(label="Select video range (second)", min_value=0, max_value=int(float(video_info['duration'])), step=10, value=(50, min(int(float(video_info['duration'])), 100)), key=f'{key}_silder')
    video_info['video_select_frame'] = video_info['video_select_range'][1] * video_info['video_fps'] - video_info['video_select_range'][0] * video_info['video_fps']
    
    url_submit = st.button(label="Submit", key=f'{key}_button')
    if url_submit or state.submit:
        if url_submit:
            state.submit = True
            
            with st.spinner("Data Preprocessing ..."):
                frames_with5 = preprocess(video_info, key=key)
            preprocess_success_msg = st.success("Data Preprocessed Successfully!")
            
            with st.spinner("Roll Data Inferencing ..."):
                roll, logit, roll_wav, pm_roll = video_to_roll_inference(video_info, frames_with5)
            roll_inference_success_msg = st.success("Data Inferenced successfully!")
            
            with st.spinner("Midi Data Inferencing ..."):
                midi, midi_wav, pm_midi = roll_to_midi_inference(video_info, logit)
            midi_inference_success_msg = st.success("Data Inferenced successfully!")
            
            state.roll, state.midi = roll, midi
            state.roll_wav, state.midi_wav = roll_wav, midi_wav
            
            time.sleep(1)
            preprocess_success_msg.empty()
            time.sleep(0.5)
            roll_inference_success_msg.empty()
            time.sleep(0.5)
            midi_inference_success_msg.empty()
        
            st.image(np.rot90(state.roll, 1), width=700)
            # st.audio(state.roll_wav, sample_rate=16000)
            # st.image(np.rot90(state.roll2, 1), width=700)
            st.audio(state.midi_wav, sample_rate=16000)
        
            with st.spinner("Generating Sheet Music..."):
                with st.expander(":musical_note: Sheet Music"):
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
                    
                    col1, col2, _, _ = st.columns(4)
                    with col1:
                        state.download_click = st.download_button(
                            label="pdf download",
                            data=open("./data/outputs/roll_sheet.pdf", 'rb').read(),
                            file_name="piano_sheet.pdf",
                            mime="application/pdf"
                        )
                    with col2:
                        state.download_click = st.download_button(
                            label="png download",
                            data=open("./data/outputs/roll_sheet.png", 'rb').read(),
                            file_name="piano_sheet.png",
                            mime="image/png"
                        )               