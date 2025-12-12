from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

MODEL_PATH = "traffic_signs_yolo.torchscript"
IMAGE_PATH = "scripts/30km.png"   # cambia por tu imagen


IMG_SIZE = 640
CONF = 0.4

model_path = Path(MODEL_PATH)
image_path = Path(IMAGE_PATH)

assert model_path.is_file(), f"No se encontró el modelo en {model_path}"
assert image_path.is_file(), f"No se encontró la imagen en {image_path}"

model = YOLO(str(model_path), task="detect")
results = model(str(image_path), imgsz=IMG_SIZE, conf=CONF, verbose=False)
r = results[0]

im_bgr = r.plot()
im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(7,7))
plt.imshow(im_rgb)
plt.axis("off")
plt.title("Detección de señales (TorchScript local)")
plt.savefig("deteccion_torchscript.png", dpi=200, bbox_inches="tight")



import os
print("Guardando en:", os.getcwd())

#Invoke-WebRequest "https://huggingface.co/spaces/Camilosss/TrafficSignDetectionYOLO/resolve/main/traffic_signs_yolo.torchscript" -OutFile "traffic_signs_yolo.torchscript"
