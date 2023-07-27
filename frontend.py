import validators
from pytube import YouTube

from process import process

import streamlit as st
from streamlit import session_state as state


# streamlit run frontend.py --server.port 30006 --server.fileWatcherType none
# https://www.youtube.com/watch?v=4VR-6AS0-l4
if __name__ == "__main__":
    
    st.set_page_config(
        page_title="Music Transcription from Silent Videos",
        page_icon="musical_keyboard")
    
    # session.state
    if "prev_url" not in state: state.prev_url = None
    if "prev_video" not in state: state.prev_video = None
    if "url_input" not in state: state.url_input = None
    if "video_input" not in state: state.video_input = None
    
    if "roll" not in state: state.roll = None
    if "roll_wav" not in state: state.roll_wav = None
    if "midi" not in state: state.midi = None
    if "midi_wav" not in state: state.midi_wav = None
    
    if "submit" not in state: state.submit = False
    if "sheet" not in state: state.sheet = False
    if "sheet_file" not in state: state.sheet_file = False

    st.header(":musical_keyboard: Music Transcription from Silent Videos")
    st.subheader("How do you want to upload the video?")
    st.markdown(
        """
        <style>
        div[class*="stTextInput"] div input {                
            height: 80px;
            padding: 1rem;
        }
        
        button[class*="css-7ym5gk"] {
            width: 130px;
        }
        
        div[class*="stCheckbox"] {
            padding-top: 7px;
        }
        
        div[class*="css-y4bq5x"] {
            width: 180px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    tab_url, tab_video = st.tabs([":link: URL", ":film_frames: VIDEO"])
        
    with tab_url:
        state.url_input = st.text_input(label="URL", placeholder="📂 Input youtube url here (ex. https://youtu.be/...)")
        
        if state.url_input:
            state.video_bytes = None
            state.sheet_file = False
            if validators.url(state.url_input):
                try:
                    if state.prev_url != state.url_input:                        
                        state.prev_url = state.url_input
                        state.submit, state.sheet = False, False
                        with st.spinner("Url Analyzing ..."):
                            yt = YouTube(state.url_input)
                            yt.streams.filter(file_extension="mp4", res="720p").order_by("resolution").desc().first().download(output_path="./data/inference", filename="01.mp4")
                except Exception as e:
                    print(e)
                    st.error("Please check Youtube url !")
                else:
                    process(key='url')
            else:
                st.error("Please input url !")
            
    with tab_video:
        state.video_input = st.file_uploader(label="VIDEO", type=["mp4", "wav", "avi"])

        if state.video_input:
            state.video_bytes = None
            state.sheet_file = False
            if state.prev_video != state.video_input:
                state.prev_video = state.video_input
                state.submit, state.sheet = False, False
                with open("./data/inference/02.mp4", "wb") as f:
                    f.write(state.video_input.getbuffer())

            process(key='video')
                