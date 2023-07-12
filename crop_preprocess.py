import sys
import numpy as np
sys.path.append('/opt/ml/final_project/ultralytics')
from ultralytics import YOLO
import cv2
import glob

def crop_piano(frame: np.ndarray, conf_threshold: float = 0.8):
    model = YOLO("./detection_models/best.pt")
    # 0번 gpu 사용
    output = model.predict(source=frame, device="0")
    xyxy = tuple()

    if output[0].boxes:
        if output[0].boxes.conf.item() > conf_threshold:
            xyxy = tuple(np.array(output[0].boxes.xyxy.detach().cpu()[0], dtype=int))

    if not xyxy:
        print("No Piano")

    return xyxy

if __name__ == "__main__":
    sample_images = glob.glob("./sample_images/*")
    img = cv2.imread(sample_images[0])[..., ::-1]
    print(crop_piano(img))

