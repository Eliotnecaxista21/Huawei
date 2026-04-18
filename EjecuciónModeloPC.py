import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# 1. Carga del modelo entrenado
model = YOLO(r'../Modelos/best.pt')

# Definicion de colores en formato BGR para la visualizacion
mis_colores = {
    "Sana": (0, 255, 0),             # Verde
    "Arana_Roja": (0, 165, 255),     # Naranja 
    "Mosca_Blanca": (255, 255, 255), # Blanco
    "Tizon_Temprano": (0, 0, 255),   # Rojo
    "Tuta_absoluta": (0, 255, 255)   # Amarillo
}

# Inicializacion de la captura de video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("--- PlagueBot: Sistema de deteccion activado ---")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Inferencia del modelo
    results = model.predict(frame, conf=0.20, imgsz=640, verbose=False)[0]
    annotator = Annotator(frame, line_width=2)
    
    for box in results.boxes:
        b = box.xyxy[0]  
        c = int(box.cls) 
        conf = float(box.conf)
        nombre = model.names[c]
        
        # Asignacion de color segun la clase detectada
        color_cuadro = mis_colores.get(nombre, (128, 128, 128)) 
            
        label = f"{nombre} {conf:.2f}"
        annotator.box_label(b, label, color=color_cuadro)

    # Visualizacion de resultados
    cv2.imshow("PlagueBot - Deteccion en Tiempo Real", annotator.result())

    # Cierre de ventana con tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberacion de recursos
cap.release()
cv2.destroyAllWindows()