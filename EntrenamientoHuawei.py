#En este codigo se descomprime el dataset descargado desde el OBS y se ejecuta una entrenamiento de 50 etapas
#Instala
#Moxing y ultralytics



# 1. Importación de librerías
import moxing as mks
import zipfile
import os
from ultralytics import YOLO

# 2. Configuración de rutas
# Ajustar según el nombre del bucket y del archivo zip en OBS
ruta_obs = 'obs://plaguebot-dataset-hk/Jada_Tomato.v1i.yolov8.zip'
ruta_local_zip = '/home/ma-user/work/dataset.zip'
ruta_extraccion = '/home/ma-user/work/dataset_tomates/'

# 3. Transferencia de datos desde OBS al entorno de cómputo (V100)
def preparar_dataset(obs_path, local_zip, extract_path):
    print("Iniciando descarga desde OBS...")
    mks.file.copy(obs_path, local_zip)
    
    print("Descomprimiendo archivos...")
    with zipfile.ZipFile(local_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Dataset listo en el entorno local.")

# 4. Corrección de dependencias (Específico para entorno ModelArts PyTorch 1.8)
# Se requiere numpy >= 1.21.6 para compatibilidad con el motor de YOLOv8
def instalar_dependencias():
    os.system('pip install numpy==1.21.6')
    os.system('pip install ultralytics')

# 5. Configuración y ejecución del entrenamiento
def ejecutar_entrenamiento(data_yaml_path):
    # Carga del modelo preentrenado (YOLOv8 nano)
    model = YOLO('yolov8n.pt')
    
    # Parámetros de entrenamiento
    model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0, 
        workers=8
    )

# Flujo principal de ejecución
if __name__ == "__main__":
    # Paso 1: Descarga y extracción
    preparar_dataset(ruta_obs, ruta_local_zip, ruta_extraccion)
    
    # Paso 2: Actualización de entorno
    instalar_dependencias()
    
    # Paso 3: Inicio del proceso de visión artificial
    # La ruta debe apuntar al archivo data.yaml generado por el dataset
    config_path = os.path.join(ruta_extraccion, 'data.yaml')
    ejecutar_entrenamiento(config_path)