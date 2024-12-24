import cv2
import numpy as np
import time
from main import load_model
from ultralytics import YOLO
from tqdm import tqdm
import torch


# Функция обработки кликов мыши
selected_id = None

def select_bbox(event, x, y, flags, param):
    global selected_id,bboxes
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print(event, x, y)
    # if event == 1:
        
        for bbox in bboxes:
            # bbox  формата [x1, y1, x2, y2, id]
            if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
                selected_id = bbox[4]  # Сохраняем выбранный ID
                print(f"Selected ID: {selected_id}")
                break


# Подключаем обработчик событий мыши
cv2.namedWindow('Tracking')
cv2.setMouseCallback('Tracking', select_bbox)


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
    if frame_number == 0:
        frame_x_c = frame.shape[1] // 2
        frame_y_c= frame.shape[0] // 2
    if not success or frame_number >= total_frame_count:
        break

    if frame_number % step_detection == 0:
        results = model.track(frame, classes = [0], persist=True, verbose=False, imgsz=640)
        bboxes = []
        for result in results:
            for box in result.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                track_id = int(box.id)
                bboxes.append([x1,y1,x2,y2, track_id])
    id_check = 0
    if selected_id is not None:
        for bbox in bboxes:
            if bbox[4] == selected_id:
                id_check = 1
                x_center = (bbox[0] + bbox[2]) // 2
                y_center = (bbox[1] + bbox[3]) // 2
                print("по x на", x_center - frame_x_c)
                print("по y на", y_center - frame_y_c)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                cv2.putText(frame, f'ID: {selected_id}', (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                # img = frame
                 
        if id_check == 0:
            selected_id = None

    elif selected_id is None:
        for bbox in bboxes:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.putText(frame, f'ID: {bbox[4]}', (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # img = frame


        # img = results[0].plot() 
        # img = results.plot()
    # else:
    #     img = frame
    cv2.imshow("Tracking", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        selected_id = None
        # key = cv2.waitKey(1)
        # if key == 27:  # esc
        #     print("esc pressed")
        #     break
    frame_number += 1

print(time.time() - st_time)
cap.release()
cv2.destroyAllWindows()