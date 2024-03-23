import logging
import configparser
from pathlib import Path
from ultralytics import YOLO
import os

def model_train():
    current_folder = os.getcwd()
    yaml_local_path = os.path.join(current_folder, 'datasets', 'data.yaml')
    num_of_epochs = 2
    image_size = 640
    model_name = 'yolov8n'

    # Load a model
    yaml_model_name = model_name + ".yaml"
    pt_model_name = model_name + ".pt"
    model = YOLO(yaml_model_name)  # build a new model from YAML
    model = YOLO(pt_model_name)  # load a pretrained model (recommended for training)
    model = YOLO(yaml_model_name).load(pt_model_name)  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=yaml_local_path, epochs=num_of_epochs, imgsz=image_size)
    print(f"Finish train model {pt_model_name}")
