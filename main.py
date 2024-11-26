from ultralytics import YOLO
import cv2
import torch
import os
import time 
import numpy as np
from webcamera import webcam
#from wasd import camera_frame
"""
засечь время 
и запустить функцию 100 раз 
"""


def load_model(model_path):

    model = YOLO(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model.to(device)
    
    return model


def find_people(model, frame, threshold=0.5):
    return model(frame, classes=0, conf=threshold)[0].plot()


def main():
    cam = frame 
    model_path = "yolo11n.pt"
    model = load_model(model_path)
    #camera_frame_main = 
    frame = cv2.imread(cam)
    print("Обработка не запустилась")
    res_func = find_people(model, frame)
    print("Работает!")
    cv2.imshow(" ", res_func)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()

