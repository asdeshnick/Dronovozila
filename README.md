# Dronovozila

### Оно умеет управлять дроном "Пионер мини", а так же детектить людей с камеры дрона

## Для работы нужна библиотека pioneer_sdk и numpy одной из старых,  и python версии 3.8версий. Конкретную верси можно в ошибке посмотреть если будет.

## есть такой касяк, но не всегда (как карта ляжет)

#### Настройка окружения

1. Создайте удобным для вас способ окружение (venv)
2. В активированном окружении запустите в терминале
`pip install ultralytics pioneer_sdk`

#### Импорт необходимых библиотек

```
from ultralytics import YOLO
import cv2
import torch
import os
import time 
import numpy as np
from pioneer_sdk import Pioneer, Camera
```

#### кофигурация 
`step_detection` шаг для взятие кадра 

