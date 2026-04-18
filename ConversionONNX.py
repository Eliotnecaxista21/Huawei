from ultralytics import YOLO

# Carga el modelo de pesos de PyTorch (.pt)
model = YOLO('yolov5nu.pt')

# Exportación a ONNX
# simplify=True ayuda a eliminar nodos redundantes antes del compilador
model.export(
    format='onnx',
    imgsz=640,
    opset=12,
    simplify=True
)