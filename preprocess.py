import cv2
import torch
import numpy as np
import concurrent.futures
import time
import warnings
warnings.filterwarnings('ignore')

def process_frame(frame, xmin, ymin, xmax, ymax):
    frame = frame[ymin:ymax, xmin:xmax]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (900, 100), interpolation=cv2.INTER_LINEAR) / 255.
    return frame

def preprocess(model, video_info, key):
    total_st = time.time()
    if key == 'url': video_path = "./data/inference/01.mp4"
    else: video_path = "./data/inference/02.mp4"
    
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)

    cap = cv2.VideoCapture(video_path)
    start, end = video_info['video_select_range'][0] * video_info['video_fps'], video_info['video_select_range'][1] * video_info['video_fps']
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    
    frames = []
    frame_count = start
    key_detected = False
    st = time.time()
    results = []
    while True:      
        if frame_count < end:
            frame_count += 1
            ret, frame = cap.read()
            if not ret:
                break
            # Piano Detection
            if not key_detected:
                pred = model.predict(source=frame, device="0", verbose=False)
                if pred[0].boxes:
                    if pred[0].boxes.conf.item() > 0.8:
                        xmin, ymin, xmax, ymax = tuple(np.array(pred[0].boxes.xyxy.detach().cpu()[0], dtype=int))
                        key_detected = True
                        future = executor.submit(process_frame, frame, xmin, ymin, xmax, ymax)
                        results.append(future)                        
                        continue
            else:               
                future = executor.submit(process_frame, frame, xmin, ymin, xmax, ymax)
                results.append(future)
        elif (not ret) or (frame_count >= end):
            break

    frames = executor.map(concurrent.futures.Future.result, results)
        
    ed = time.time()
    print(f"time to video read : {ed-st:.4f} s")
    executor.shutdown()

    st = time.time()
    frames = np.stack(frames, axis=0)
    ed = time.time()
    print(f"time to stack : {ed-st:.4f} s")

    # 5 frame 씩
    st = time.time()
    frames_with5 = []
    for i in range(len(frames)):
        if i >= 2 and i < len(frames)-2:
            file_list = [frames[i-2], frames[i-1], frames[i], frames[i+1], frames[i+2]]
        elif i < 2:
            file_list = [frames[i], frames[i], frames[i], frames[i+1], frames[i+2]]
        else:
            file_list = [frames[i-2], frames[i-1], frames[i], frames[i], frames[i]]
        frames_with5.append(file_list)
    ed = time.time()
    print(f"time to making chunks : {ed-st:.4f} s")

    
    total_ed = time.time()
    print(f"time to total : {total_ed-total_st:.4f} s")
    return frames_with5