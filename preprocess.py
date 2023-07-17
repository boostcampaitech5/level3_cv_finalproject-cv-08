import cv2
import torch
import numpy as np


def preprocess(model, video_info, key):
    if key == 'url': video_path = "./data/inference/01.mp4"
    else: video_path = "./data/inference/02.mp4"
    
    cap = cv2.VideoCapture(video_path)
    start, end = video_info['video_select_range'][0] * video_info['video_fps'], video_info['video_select_range'][1] * video_info['video_fps']
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    
    frames = []
    frame_count = 0
    key_detected = False
    
    while True:
        if frame_count < start: 
            frame_count += 1
        elif start <= frame_count < end:
            frame_count += 1
            ret, frame = cap.read()
            
            # Piano Detection
            if not key_detected:
                pred = model.predict(source=frame, device="0", verbose=False)
                if pred[0].boxes:
                    if pred[0].boxes.conf.item() > 0.8:
                        xmin, ymin, xmax, ymax = tuple(np.array(pred[0].boxes.xyxy.detach().cpu()[0], dtype=int))
                        key_detected = True
                        continue
            else:               
                frame = frame[ymin:ymax, xmin:xmax]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (900, 100), interpolation=cv2.INTER_LINEAR)
                # cv2.imwrite("02.jpg", frame)
                frames.append(frame / 255.)
        elif (not ret) or (frame_count >= end):
            break
    
    frames = np.stack(frames, axis=0)

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
    
    return frames_with5
