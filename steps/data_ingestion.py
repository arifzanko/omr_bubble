import logging
import boto3
import os
import configparser
from pathlib import Path
import yaml
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO, format='%(message)s')
confp = configparser.RawConfigParser()
confp.read(os.path.abspath(os.path.join(Path(__file__).absolute(), os.pardir)) + '/config.ini')

aws_access_key_id = confp.get("aws","aws_access_key_id")
aws_secret_access_key = confp.get("aws","aws_secret_access_key")
bucket_name = confp.get("s3","bucket")
datasets_path_s3 = confp.get("s3", "datasets_path")
datasets_folder_local = confp.get("local", "datasets_local_path")


class IngestData:
    """
    Ingesting data from S3
    """
    def __init__(self, datasets_path_s3: str, aws_access_key_id: str, aws_secret_access_key: str):
        self.datasets_path_s3 = datasets_path_s3
        self.s3_session = self.get_s3_connection(aws_access_key_id, aws_secret_access_key)

    def create_folder(self):
        """
        Create folder for store datasets
        """
        current_folder = os.getcwd()
        datasets_path_local = os.path.join(current_folder, datasets_folder_local)
        if not os.path.exists(datasets_path_local):
            os.makedirs(datasets_path_local)
            logging.info(f"Folder {datasets_folder_local} created")
        
        # Create subfolders train, test, and valid
        subfolders = ['train', 'test', 'valid']
        for subfolder in subfolders:
            subfolder_path = os.path.join(datasets_path_local, subfolder)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
                logging.info(f"Subfolder {subfolder} created inside {datasets_folder_local}")
        
            # Create 'images' folder
            images_folder_path = os.path.join(subfolder_path, 'images')
            if not os.path.exists(images_folder_path):
                os.makedirs(images_folder_path)
            
            # Create 'labels' folder
            labels_folder_path = os.path.join(subfolder_path, 'labels')
            if not os.path.exists(labels_folder_path):
                os.makedirs(labels_folder_path)

        # Create data.yaml file
        data_yaml_path = os.path.join(datasets_path_local, 'data.yaml')
        if not os.path.exists(data_yaml_path):
            data_yaml_content = {}  # You can customize the content as needed
            with open(data_yaml_path, 'w') as yaml_file:
                yaml.dump(data_yaml_content, yaml_file)
                logging.info(f"data.yaml file created inside {datasets_folder_local}")
    
    def create_temp_datasets_folder(self):
        """
        Create a temporary folder with subfolders and empty files
        """
        current_folder = os.getcwd()
        datasets_path_local = os.path.join(current_folder, 'datasets_temp')
        images_path = os.path.join(datasets_path_local, 'images')
        labels_path = os.path.join(datasets_path_local, 'labels')
        note_file_path = os.path.join(datasets_path_local, 'notes.json')
        classes_file_path = os.path.join(datasets_path_local, 'classes.txt')

        # Create datasets_temp folder
        if not os.path.exists(datasets_path_local):
            os.makedirs(datasets_path_local)
            logging.info(f"Folder {datasets_path_local} created")

        # Create images and labels subfolders
        for subfolder_path in [images_path, labels_path]:
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
                logging.info(f"Subfolder {subfolder_path} created")

        # Create empty note.json file
        if not os.path.exists(note_file_path):
            with open(note_file_path, 'w') as note_file:
                note_data = {}
                json.dump(note_data, note_file)
            logging.info(f"File {note_file_path} created")

        # Create empty classes.txt file
        if not os.path.exists(classes_file_path):
            with open(classes_file_path, 'w') as classes_file:
                pass  # Creates an empty file
            logging.info(f"File {classes_file_path} created")

        print("Temporary folder and files created successfully.")


    def get_s3_connection(self, aws_access_key_id, aws_secret_access_key):
        s3_session = boto3.client('s3', aws_access_key_id = aws_access_key_id, aws_secret_access_key = aws_secret_access_key)
        return s3_session

    def download_images(self):
        s3_path_images = self.datasets_path_s3 + '/images/'
        current_folder = os.getcwd()
        folder_path_images = os.path.join(current_folder, "datasets_temp/images")
        print(folder_path_images, bucket_name, s3_path_images)

        objects = self.s3_session.list_objects(Bucket=bucket_name, Prefix=s3_path_images)['Contents']
        for index, obj in enumerate(objects):
            if index == 0:
                continue  # Skip the first iteration
            key = obj['Key']
            local_file_path = os.path.join(folder_path_images, os.path.basename(key))
            self.s3_session.download_file(bucket_name, key, local_file_path)
            # print(f'Downloaded: {key} to {local_file_path}')
        print(f"Finish download images {folder_path_images}")

    def download_labels(self):
        s3_path_images = self.datasets_path_s3 + "/labels/"
        current_folder = os.getcwd()
        folder_path_labels = os.path.join(current_folder, "datasets_temp/labels")
        print(folder_path_labels, bucket_name, s3_path_images)

        objects = self.s3_session.list_objects(Bucket=bucket_name, Prefix=s3_path_images)['Contents']
        for index, obj in enumerate(objects):
            if index == 0:
                continue  # Skip the first iteration
            key = obj['Key']
            local_file_path = os.path.join(folder_path_labels, os.path.basename(key))
            self.s3_session.download_file(bucket_name, key, local_file_path)
            # print(f'Downloaded: {key} to {local_file_path}')
        print(f"Finish download labels {folder_path_labels}")

    def get_json_file(self):
        """
        Ingesting the json file from the datasets_path
        """
        json_s3_path = self.datasets_path_s3 + '/notes.json'
        current_folder = os.getcwd()
        json_local_path = os.path.join(current_folder, 'datasets_temp/notes.json')

        s3 = boto3.client('s3', aws_access_key_id = aws_access_key_id, aws_secret_access_key = aws_secret_access_key)
        objects = s3.list_objects(Bucket=bucket_name, Prefix=json_s3_path)['Contents']
        s3.download_file(bucket_name, json_s3_path, json_local_path)
        logging.info(f"Finish download json file {json_local_path}")

    def get_txt_file(self):
        """
        Ingesting the txt file from the datasets_path
        """
        txt_s3_path = self.datasets_path_s3 + '/classes.txt'
        current_folder = os.getcwd()
        txt_local_path = os.path.join(current_folder, 'datasets_temp/classes.txt')

        s3 = boto3.client('s3', aws_access_key_id = aws_access_key_id, aws_secret_access_key = aws_secret_access_key)
        objects = s3.list_objects(Bucket=bucket_name, Prefix=txt_s3_path)['Contents']
        s3.download_file(bucket_name, txt_s3_path, txt_local_path)
        logging.info(f"Finish download txt labels {txt_local_path}")


def ingest_df():
    ingest_data = IngestData(datasets_path_s3, aws_access_key_id, aws_secret_access_key)
    ingest_data.create_folder()
    ingest_data.create_temp_datasets_folder()
    ingest_data.download_images()
    ingest_data.download_labels()
    ingest_data.get_json_file()
    ingest_data.get_txt_file()