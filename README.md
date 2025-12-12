# 🚦 Detección de Señales de Tránsito con YOLOv8

Proyecto final de la asignatura **Procesamiento Digital de Imágenes (2025-2)**.  
El objetivo es entrenar y desplegar un modelo de **detección de objetos** capaz de localizar y clasificar diferentes **señales de tránsito** en imágenes.

> **Autores:** Juan Camilo Miño Castillo, Luis Eduardo Miño Castillo
> **Profesor:** Lucas Miguel Iturriago Salas 
> **Curso:** PDI – 2025-2

---

## 1. Descripción del proyecto

En este proyecto se entrena un modelo **YOLOv8** para detectar distintas señales de tránsito (por ejemplo límites de velocidad, señales de prohibición, etc.) en imágenes RGB.

El pipeline completo incluye:

1. **Preparación del dataset** desde Roboflow (formato YOLOv8).
2. **Entrenamiento** del modelo en Google Colab.
3. **Evaluación** del desempeño (mAP, matriz de confusión, curva F1–confianza).
4. **Exportación del modelo** a formato **TorchScript**.
5. **Despliegue** del modelo en un **HuggingFace Space** mediante FastAPI + Docker.
6. **Inferencia local** en Python (carga directa del modelo).
7. **Inferencia remota vía API** consumiendo el Space de HuggingFace.

---

## 2. Dataset

- **Origen:** Roboflow  
- **Tarea:** Detección de objetos (YOLOv8)  
- **Número de imágenes:** = 4.720  
- **Split:** train / valid / test  
- **Preprocesamiento principal:**
  - Resize a **512×512**
  - Auto-orientación de imágenes

- **Clases (ejemplos):**
  - Límites de velocidad (10, 20, 30, 40, 60, …)
  - Señal de prohibición (stop)
  - Color del semáforo (verde, rojo)

👉 **Enlace al dataset (Roboflow):**  
`https://app.roboflow.com/universidad-nacional-o6onq/detect-project-lqv4y/models` 

---

## 3. Estructura del repositorio

```text
.
├── notebooks/
│   └── modelo  # Notebook principal: entrenamiento + métricas + TorchScript + ejemplos  Modelo exportado a TorchScript + HuggingFace
├── scripts/
│   └── inferencia.ipynb                    # Inferencia consumiendo la API en HuggingFace
├── recursos/ # Videos e imagenes utilizadas para probar el modelo
│
├── PDI_PRESENTACION_FINAL.pdf
├── requirements.txt                    # Dependencias de Python
└── README.md
````

---

## 4. Requisitos

* Python 3.10+
* Paquetes principales:

  * `ultralytics`
  * `torch`, `torchvision`
  * `opencv-python`
  * `matplotlib`
  * `fastapi`, `uvicorn` (para la API)
  * `requests`

Instalación rápida:

```bash
pip install -r requirements.txt
```

Si solo quieres probar **inferencia local**:

```bash
pip install ultralytics opencv-python matplotlib
```

---

## 5. Entrenamiento del modelo (YOLOv8)

El entrenamiento se realiza en el notebook:

* `notebooks/modelo.ipynb` 
Pasos principales en el notebook:

1. Descargar el dataset desde Roboflow (formato YOLOv8).
2. Definir hiperparámetros:

   * Modelo base (por ejemplo `yolov8n.yaml`)
   * `imgsz`, `epochs`, `batch`, etc.
3. Entrenar:

   * `model.train(data=data_yaml, ...)`
4. Visualizar resultados:

   * Pérdidas de entrenamiento.
   * mAP50 y mAP50-95.
   * Ejemplos de detección sobre imágenes de validación.

El notebook genera automáticamente:

* `results.csv` con la evolución de métricas.
* `confusion_matrix.png`
* `F1_curve.png`

---

## 6. Exportación a TorchScript

En el mismo notebook se exporta el modelo entrenado a **TorchScript** usando Ultralytics:

```python
from ultralytics import YOLO

model = YOLO("ruta/al/best.pt")
exported_file = model.export(
    format="torchscript",
    imgsz=640,
    optimize=False   # para evitar problemas con xnnpack
)
```

Esto genera un archivo tipo:

```text
signs_detection/yolov8n_signsX/weights/best.torchscript
```

Ese archivo se copia como:

```text
models/traffic_signs_yolo.torchscript
```

Además, en el notebook se comparan:

* Tiempos de inferencia del modelo `.pt` vs `.torchscript`.
* Resultados visuales sobre la misma imagen.

---

## 7. Inferencia local (script `infer_local.py`)

Este script carga el modelo YOLO (`.pt` o `.torchscript`) desde disco, ejecuta la detección sobre una imagen y guarda el resultado con las cajas dibujadas.

Uso:

```bash
python scripts/infer_local.py \
    --model models/traffic_signs_yolo.torchscript \
    --image data/ejemplo.jpg \
    --conf 0.4 \
    --imgsz 640 \
    --output outputs/local_prediction.jpg
```

Parámetros:

* `--model`: ruta al modelo entrenado (`.pt` o `.torchscript`).
* `--image`: ruta a la imagen de entrada.
* `--conf`: umbral de confianza.
* `--imgsz`: tamaño de la imagen de entrada.
* `--output`: ruta donde se guardará la imagen con las cajas.

---

## 8. Despliegue en HuggingFace Space

El modelo se despliega en un **Space** de tipo **Docker** usando FastAPI.

* **Space:** `Camilosss/TrafficSignDetectionYOLO`
* **URL pública:**
  `https://camilosss-trafficsigndetectionyolo.hf.space`

Dentro del Space se incluyen los archivos:

* `app.py` – API REST (FastAPI) con endpoints:

  * `GET /` – mensaje de bienvenida.
  * `GET /health` – estado del modelo.
  * `POST /predict` – recibe una imagen en base64 y devuelve detecciones en JSON.
* `traffic_signs_yolo.torchscript` – modelo exportado.
* `requirements.txt`, `runtime.txt`, `Dockerfile`, `README.md`.

Ejemplo de respuesta de `/predict`:

```json
{
  "num_detections": 1,
  "detections": [
    {
      "class_id": 5,
      "class_name": "30",
      "confidence": 0.94,
      "x1": 177.2,
      "y1": 16.3,
      "x2": 406.2,
      "y2": 243.4
    }
  ],
  "image_size": [315, 474]
}
```

---

## 9. Inferencia vía API (script `infer_api.py`)

Este script:

1. Lee una imagen local.
2. La codifica en **base64**.
3. Envía un `POST` al endpoint `/predict` del Space.
4. Imprime el JSON de salida.
5. Dibuja las cajas en la imagen y la guarda en disco.

Uso:

```bash
python scripts/infer_api.py \
    --image data/ejemplo.jpg \
    --url https://camilosss-trafficsigndetectionyolo.hf.space/predict \
    --conf 0.4 \
    --output outputs/api_prediction.jpg
```

Parámetros:

* `--image`: ruta a la imagen local.
* `--url`: URL del endpoint `/predict` del Space.
* `--conf`: umbral de confianza.
* `--output`: ruta de la imagen con cajas dibujadas.

---

## 10. Resultados y métricas

En el notebook se reportan:

* **mAP50** y **mAP50-95** sobre el conjunto de validación.
* **Matriz de confusión** normalizada por clase.
* **F1–Confidence curve** indicando el umbral óptimo de confianza.

Además, se muestran:

* Ejemplos de detección sobre imágenes de validación.
* Ejemplo de detección local (modelo TorchScript).
* Ejemplo de detección vía API (Space de HuggingFace).

---

## 11. Limitaciones y trabajo futuro

* El modelo se entrenó con imágenes principalmente bien iluminadas; podría fallar con:

  * Escenas nocturnas.
  * Señales parcialmente tapadas.
  * Señales muy lejanas o desenfocadas.
* No se ha optimizado todavía para correr en tiempo real en un sistema embebido (Raspberry Pi, etc.).
* Podría ampliarse:

  * Añadiendo nuevas clases de señales.
  * Probando modelos YOLO más grandes (`yolov8s`, `yolov8m`) para mayor precisión.
  * Integrando seguimiento de objetos (tracking) en video.

---

## 12. Cómo reproducir (resumen rápido)

1. **Clonar el repo** y crear entorno:

```bash
git clone https://github.com/<usuario>/<repo>.git
cd <repo>
pip install -r requirements.txt
```

2. **Entrenar / revisar entrenamiento**
   Abrir `notebooks/01_yolov8_traffic_signs.ipynb` en Colab o Jupyter y ejecutar.

3. **Inferencia local con imagen:**

```bash
python scripts/infer_local.py --model models/traffic_signs_yolo.torchscript --image data/ejemplo.jpg
```

4. **Inferencia vía API (Space HF):**

```bash
python scripts/infer_api.py --image data/ejemplo.jpg
```

---

Cualquier duda o mejora futura (nuevas clases, otros modelos, despliegue en dispositivos embebidos) se puede documentar en issues o forks del repositorio.

