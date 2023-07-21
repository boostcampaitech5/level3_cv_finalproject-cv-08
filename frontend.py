import validators
from pytube import YouTube

from process import process

import streamlit as st
from streamlit import session_state as state


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
    if "midi" not in state: state.midi = None
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
                    process(key='url')
            else:
                st.error("Please input url !")
            
    with tab_video:
        input_video = st.file_uploader(label="VIDEO", type=["mp4", "wav", "avi"])

        if input_video:
            with open("./data/inference/02.mp4", "wb") as f:
                f.write(input_video.getbuffer())

            process(key='video')
                