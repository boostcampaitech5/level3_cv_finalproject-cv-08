import cv2
import torch
import ffmpeg
import numpy as np


def preprocess(model, video_info, key):
    if key == 'url': video_path = "./data/inference/01.mp4"
    else: video_path = "./data/inference/02.mp4"
    out, _ = (
        ffmpeg
        .input(video_path, ss=video_info['video_select_range'][0], t=video_info['video_select_range'][1]-video_info['video_select_range'][0])
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', loglevel="quiet")
        .run(capture_stdout=True)
    )
    
    frames = (
        np
        .frombuffer(out, np.uint8)
        .reshape([-1, video_info['height'], video_info['width'], 3])
    )

    key_detected = False
    for i, frame in enumerate(frames):
        # Piano Detection
        if not key_detected:
            pred = model.predict(source=frame, device='0', verbose=False)
            if pred[0].boxes:
                if pred[0].boxes.conf.item() > 0.8:
                    xmin, ymin, xmax, ymax = tuple(np.array(pred[0].boxes.xyxy.detach().cpu()[0], dtype=int))
                    cv2.imwrite("02.jpg", cv2.cvtColor(frame[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2GRAY))
                    start_idx = i
                    key_detected = True
                    break
    
    if key_detected:
        frames = np.mean(frames[start_idx:, ymin:ymax, xmin:xmax, ...], axis=3)
        frames = np.stack([cv2.resize(f, (900 ,100), interpolation=cv2.INTER_LINEAR) for f in frames], axis=0) / 255.
    else:
        # 영상 전체에서 Top View 피아노가 없을 경우 None 반환
        return None
        
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
