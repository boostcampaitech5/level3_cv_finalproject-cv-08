import io
import json
import base64
import requests
import numpy as np
from PIL import Image

import streamlit as st
from streamlit import session_state as state

st.set_page_config(page_title="Piano2Roll")

def main():
    # session.state
    if "tab_url" not in state: state.tab_url = None
    if "tab_video" not in state: state.tab_video = None

    if "input_url" not in state: state.url = None
    if "video" not in state: state.video = None
    
    if "frame_range" not in state: state.frame_range = None
    
    st.header("Inference")    

    # text_input set
    st.markdown(
        """
        <style>
        div[class*="stTextInput"] div input {                
            height: 80px;
            padding: 1rem;
        }
        </style>         
        """, unsafe_allow_html=True)

    # https://youtu.be/_3qnL9ddHuw
    state.input_url = st.text_input(label="URL", placeholder="📂 Input youtube url here (ex. https://youtu.be/...)")

    input_url = {"url": state.input_url}
    
    if state.input_url:
        response = requests.post(url="http://127.0.0.1:30006/url", data=json.dumps(input_url))
        
        if not response.ok:
            st.error("Please input Youtube url (2)!")
        else:
            if response.status_code == 200:
                video_info = response.json()
                video_range = st.slider(label="Select video range (second)", min_value=0, max_value=int(float(video_info['duration'])), step=10, value=(50, 100))
                
                video_info_submit = st.button(label="Submit", key="video_info_submit")
                if video_info_submit:
                    input_video_info = {"video_info": video_info, "video_range": video_range}
                    response = requests.post(url="http://127.0.0.1:30006/submit", data=json.dumps(input_video_info))
                    
                    """ 
                    base64로 encode된 결과 수신
                    decode 후 출력
                    """
                    result = json.loads(response.json())
                    
                    encoded_roll = result['roll']
                    decoded_roll_data = base64.b64decode(encoded_roll)
                    roll = Image.open(io.BytesIO(decoded_roll_data))
                    
                    encoded_wav = result['wav']
                    decoded_wav_data = base64.b64decode(encoded_wav)
                    wav = Image.open(io.BytesIO(decoded_wav_data))
                    
                    st.image(roll, width=700)
                    st.audio(wav, sample_rate=16000)

 
if __name__ == '__main__':
    main()