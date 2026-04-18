# PlagueBot
### Eliot Calderón Romero

## Objective
The detection of common tomato field pests through a vision system implemented on a robot to be deployed in the field. The system utilizes Edge AI technology to process video locally on the robot, eliminating internet dependency.

## Dataset
We compiled several online datasets, selecting only the data necessary to detect common tomato pests. Additionally, we gathered web images and performed manual labeling and bounding box annotation. These efforts, combined with data augmentation techniques (including rotation, blur, and high contrast), resulted in a robust dataset of 19,000 images. This dataset is hosted on Roboflow as Jada_Tomato.

## Training
Various training platforms were tested to identify the optimal choice regarding performance and workflow efficiency. Tests were conducted on Roboflow, Kaggle, Google Colab, ModelArts, and a local RTX 3050 laptop GPU.

ModelArts significantly outperformed its competitors. It benefits from the Huawei ecosystem, which greatly streamlines the training process when using its integrated technologies. Its training speed was notably superior: using the available GPU, it completed a 50-epoch training session in just over an hour, while other platforms required nearly two hours for only 30 epochs—a substantial difference in efficiency.

## Models
The YOLO architecture was selected due to its ease of deployment on hardware accelerators. The implementation uses a Raspberry Pi equipped with a Hailo-8 AI kit (26 TOPS), enabling high-speed model analysis reaching up to 90 FPS. To achieve this, the YOLO .pt file is converted to ONNX and subsequently to a Hailo-compatible format for implementation on the Raspberry Pi.

Regarding the model generated with ModelArts, it is possible to obtain a MindSpore file for native execution on Ascend chips. However, since we did not have access to this specific hardware, testing for that platform could not be conducted.

## Detection Classes
The model is trained to identify the following 5 categories:
1. Healthy (Sana)
2. Red Spider Mite (Araña Roja)
3. Whitefly (Mosca Blanca)
4. Early Blight (Tizón Temprano)
5. Tomato Leafminer (Tuta absoluta)


## Performance Metrics
* **Inference Speed:** Up to 90 FPS on Raspberry Pi + Hailo-8.
* **Input Resolution:** 640x640 pixels.
* **Quantization:** INT8 (Post-Training Quantization).
