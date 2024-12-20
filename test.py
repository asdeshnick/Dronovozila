import cv2
import numpy as np
import time
from main import find_people, load_model
from ultralytics import YOLO
from tqdm import tqdm

model = load_model("yolo11n.pt")

video_path = "test_images/people_walking.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)


frame_number = 0
step_detection = 9
class_name = 0

total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(total_frame_count)
st_time = time.time()
# for frame_number in tqdm(range(total_frame_count), desc='Обработка кадров для ящиков'):
while True:
    success, frame = cap.read()
    if not success or frame_number >= total_frame_count:
        break
    if frame_number % step_detection == 0:

        results = model.track(frame, classes = [0], persist=True, verbose=False, imgsz=480)[0]

        img = results.plot()
        
        cv2.imshow("asd", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            # key = cv2.waitKey(1)
            # if key == 27:  # esc
            #     print("esc pressed")
            #     break
    frame_number += 1

print(time.time() - st_time)
cap.release()
cv2.destroyAllWindows()