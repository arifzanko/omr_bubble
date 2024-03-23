import mlflow
from typing import Any
import os
from pathlib import Path
import configparser
import logging
import os
import csv

confp = configparser.RawConfigParser()
confp.read(os.path.abspath(os.path.join(Path(__file__).absolute(), os.pardir)) + '/config.ini')
num_of_epochs = confp.get("train","num_of_epochs")
image_size = confp.get("train","image_size")
artifact_path = confp.get("mlflow", "artifact_path")
model_name = confp.get("train", "model")

def create_mlflow_experiment(experiment_name: str, artifact_location: str, tags:dict[str,Any]) -> str:
    """
    Create a new mlflow experiment with the given name and artifact location.
    """

    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_location, tags=tags
        )
    except:
        logging.info(f"Experiment {experiment_name} already exists.")
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    return experiment_id


def get_mlflow_experiment(experiment_id:str=None, experiment_name:str=None) -> mlflow.entities.Experiment:
    """
    Retrieve the mlflow experiment with the given id or name.

    Parameter:
    -----------
    experiment_id: str
        The id of the experiment to retrieve.
    experiment_name: str
        The name of the experiment to retrieve.

    Returns:
    -----------
    experiment: mlflow.entities.Experiment
        The mlflow experiment with the given id or name.
    """
    if experiment_id is not None:
        experiment = mlflow.get_experiment(experiment_id)
    elif experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    else:
        raise ValueError("Either experiment_id or experiment_name must be provided.")
    return experiment

def get_metrics():
    current_path = os.getcwd()
    #train_output_path = "runs/detect/train/results.csv"

    last_row = None
    csv_file_path = os.path.join(current_path, 'runs', 'detect', 'train', 'results.csv')
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            last_row = row

    train_box_loss = float(last_row[1].replace(" ",""))
    train_cls_loss = float(last_row[2].replace(" ", ""))
    train_dfl_loss = float(last_row[3].replace(" ", ""))
    metrics_precision_B = float(last_row[4].replace(" ", ""))
    metrics_recall_B = float(last_row[5].replace(" ", ""))
    metrics_mAP50_B = float(last_row[6].replace(" ", ""))
    metrics_mAP50_95_B = float(last_row[7].replace(" ", ""))
    val_box_loss = float(last_row[8].replace(" ", ""))
    val_cls_loss = float(last_row[9].replace(" ", ""))
    val_dfl_loss = float(last_row[10].replace(" ", ""))
    lr_pg0 = float(last_row[11].replace(" ", ""))
    lr_pg1 = float(last_row[12].replace(" ", ""))
    lr_pg2 = float(last_row[13].replace(" ", ""))
    
    metrics = {
    "train_box_loss": train_box_loss,
    "train_cls_loss": train_cls_loss,
    "train_dfl_loss": train_dfl_loss,
    "metrics_precision_B" : metrics_precision_B,
    "metrics_recall_B" : metrics_recall_B,
    "metrics_mAP50_B" : metrics_mAP50_B,
    "metrics_mAP50_95_B" : metrics_mAP50_95_B,
    "val_box_loss" : val_box_loss,
    "val_cls_loss" : val_cls_loss,
    "val_dfl_loss" : val_dfl_loss,
    "lr_pg0" : lr_pg0,
    "lr_pg1" : lr_pg1,
    "lr_pg2" : lr_pg2,
    }

    return metrics

def get_artifacts(artifact_path):
    current_path = os.getcwd()
    complete_artifact_path = os.path.join(current_path, artifact_path)
    file_names = [
                  "args.yaml", 
                  "confusion_matrix_normalized.png", 
                  "confusion_matrix.png", 
                  "F1_curve.png", 
                  "labels_correlogram.jpg", 
                  "labels.jpg", 
                  "P_curve.png", 
                  "PR_curve.png", 
                  "R_curve.png", 
                  "results.csv",
                  "results.png",
                  "train_batch0.jpg",
                  "train_batch1.jpg",
                  "train_batch2.jpg",
                  "val_batch0_labels.jpg",
                  "val_batch0_pred.jpg"
                  ]

    for file_name in file_names:
        file_path = os.path.join(complete_artifact_path, file_name)
        mlflow.log_artifact(local_path=file_path)

    best_weights_path = os.path.join(complete_artifact_path, "weights", "best.pt")
    last_weights_path = os.path.join(complete_artifact_path, "weights", "last.pt")
    mlflow.log_artifact(local_path=best_weights_path, artifact_path="weights")
    mlflow.log_artifact(local_path=last_weights_path, artifact_path="weights")

def get_data_analysis():
    current_path = os.getcwd()
    data_test_path = os.path.join(current_path, "data_test_analysis.png")
    data_train_path = os.path.join(current_path, "data_train_analysis.png")
    data_valid_path = os.path.join(current_path, "data_valid_analysis.png")

    mlflow.log_artifact(local_path=data_test_path, artifact_path="datasets_analysis")
    mlflow.log_artifact(local_path=data_train_path, artifact_path="datasets_analysis")
    mlflow.log_artifact(local_path=data_valid_path, artifact_path="datasets_analysis")

def model_track():
    experiment_id = create_mlflow_experiment(
                                            experiment_name="omr",
                                            artifact_location="omr_artifacts",
                                            tags={"env":"dev", "version":"1.0.0"},
                                            )
    experiment = get_mlflow_experiment(experiment_id=experiment_id)
    with mlflow.start_run(run_name="testing", experiment_id=experiment.experiment_id) as run:

        parameters = {
            "model": model_name,
            "epochs": num_of_epochs,
            "image_size": image_size,
        }
        mlflow.log_params(parameters)
        logging.info("Parameters logged.")

        metrics = get_metrics()
        mlflow.log_metrics(metrics)
        logging.info("Metrics logged.")

        get_artifacts(artifact_path)
        logging.info("Artifacts logged.")
        #get_data_analysis()
        logging.info("Data analysis logged.")
        logging.info("run_id: {}".format(run.info.run_id))
