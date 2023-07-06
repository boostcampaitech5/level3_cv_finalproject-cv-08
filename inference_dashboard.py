import os
import glob
import numpy as np
import Video2RollNet
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

import streamlit as st
from streamlit import session_state as state

def load_model():
    model_path = "./models/Video2RollNet.pth"
    
    device = torch.device('cuda')
    with st.spinner("Loading model ..."):
        state.model = Video2RollNet.resnet18().to(device)
        state.model.load_state_dict(torch.load(model_path, map_location=device))
        state.model.eval()
        
    st.success("Model loaded successfully!")


def load_data(img_files=None):
    with st.spinner("Loading data ..."):
        # img_files : 전처리가 끝난 후 나온 image 파일 list
        img_files = glob.glob(os.path.join("./images/inference/0", "*.png"))
        img_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

        # 5 frame 씩
        files = []
        for i in range(len(img_files)):
            if i >= 2 and i < len(img_files)-2:
                # i=2 : 000001.png ~ 000005.png
                file_list = [img_files[i-2], img_files[i-1], img_files[i], img_files[i+1], img_files[i+2]]
            elif i < 2:
                file_list = [img_files[i], img_files[i], img_files[i], img_files[i+1], img_files[i+2]]
            else:
                file_list = [img_files[i-2], img_files[i-1], img_files[i], img_files[i], img_files[i]]            
            files.append(file_list)
    
    st.success("Data loaded successfully!") 
    return files


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


def inference(files, frame_range):
    save_dir_root = "./results"
    save_dir = os.path.join(save_dir_root, "0")
    os.makedirs(save_dir, exist_ok=True)
    
    min_key, max_key = 15, 65
    threshold = 0.4

    with st.spinner("Loading data ..."):
        start, end = frame_range[0]*25, frame_range[1]*25
        save_est_logit, save_est_roll = [], []
        
        inference_data = files[start:end]
        for frame in inference_data:
            torch_input_img = torch_preprocess(frame)            
            pred_logits = state.model(torch.unsqueeze(torch_input_img, dim=0))
            pred_label = torch.sigmoid(pred_logits) >= threshold            
            numpy_pred_logit = pred_logits.cpu().detach().numpy()
            numpy_pred_label = pred_label.cpu().detach().numpy().astype(np.int_)
            save_est_logit.append(numpy_pred_logit)
            save_est_roll.append(numpy_pred_label)
            
        # Roll prediction
        roll = np.zeros((end-start, 88))
        roll[:, min_key:max_key+1] = np.asarray(save_est_roll).squeeze()
        save_est_roll = roll
        
        # Logit
        target_ = np.zeros((end-start, 88))
        target_[:, min_key:max_key + 1] = np.asarray(save_est_logit).squeeze()
        save_est_logit = target_
        
        # save both Roll predictions and logits as npz files
        # np.savez(f'{save_dir}/' + str(start) + '-' + str(end) + '.npz', logit=save_est_logit, roll=save_est_roll)

    st.success("Inferenced successfully!")
    return roll

def preprocessing(_file):
    pass

# streamlit run /opt/ml/Audeo/inference_dashboard.py --server.port 30006
if __name__ == "__main__":
    st.set_page_config(page_title="Piano To Roll")
    
    if "model" not in state: state.model = None
    if "num_frame" not in state: state.num_frame = None
    
    st.header("Inference")
    
    st.subheader("How to upload ?")
    
    tab_url, tab_file = st.tabs([":link: URL", ":film_frames: VIDEO"])
    
    with tab_url:
        _url = st.text_input(label="URL", placeholder="Input youtube url here")
        
        if _url:
            inference(_url)

    with tab_file:
        _file = st.file_uploader(label="VIDEO", type=["mp4", "wav", "avi"])

        if _file:
            load_model()
            img_files = preprocessing(_file)
            files = load_data(img_files)
            state.num_frame = len(files)
            
            frame_range = st.select_slider(label="Select video range (second)", options=[i for i in range(1, (state.num_frame+1)//25)], value=(60, 120))
            _submit = st.button(label="Submit")
            
            if _submit:
                roll = inference(files, frame_range)
                st.image(np.rot90(roll, 1), width=720)
