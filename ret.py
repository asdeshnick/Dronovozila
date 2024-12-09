from flask import Flask, request, jsonify
import numpy as np
import cv2
from ultralytics import YOLO  # Вставьте вашу библиотеку для работы с моделью

app = Flask(__name__)

# Загрузка модели
model = YOLO("yolo11n.pt")  # Укажите путь к модели

@app.route('/inference', methods=['POST'])
def inference():
    # if 'file' not in request.files:
    #     return jsonify({'error': 'No file part'}), 400
    
    # file = request.files['file']
    
    # if file.filename == '':
    #     return jsonify({'error': 'No selected file'}), 400
    
    # # Прочитать содержимое файла
    # file_bytes = file.read()
    
    # # Конвертируем байты в массив NumPy
    # np_arr = np.frombuffer(file_bytes, np.uint8)
    # image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Декодируем изображение
    
    # Выполнение инференса
    results = model.predict(
        "test_images/image_2024-11-21_16-57-53.png",
        save=True,
        imgsz=320,
        conf=0.5
        )[0].boxes.xyxy.tolist()  # Ваша функция инференса
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)