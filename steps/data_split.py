import os
import shutil
import random
import yaml
import json


def split_dataset(images_folder, labels_folder, train_ratio=0.8, test_ratio=0.1, valid_ratio=0.1):
    # Create train, test, and valid folders
    train_folder = "datasets/train"
    test_folder = "datasets/test"
    valid_folder = "datasets/valid"

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(valid_folder, exist_ok=True)

    # Create subfolders for images and labels in train folder
    train_images_folder = os.path.join(train_folder, "images")
    train_labels_folder = os.path.join(train_folder, "labels")
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(train_labels_folder, exist_ok=True)

    # Create subfolders for images and labels in test folder
    test_images_folder = os.path.join(test_folder, "images")
    test_labels_folder = os.path.join(test_folder, "labels")
    os.makedirs(test_images_folder, exist_ok=True)
    os.makedirs(test_labels_folder, exist_ok=True)

    # Create subfolders for images and labels in valid folder
    valid_images_folder = os.path.join(valid_folder, "images")
    valid_labels_folder = os.path.join(valid_folder, "labels")
    os.makedirs(valid_images_folder, exist_ok=True)
    os.makedirs(valid_labels_folder, exist_ok=True)

    # Get the list of image files
    image_files = [f for f in os.listdir(images_folder) if f.endswith(".jpg")]

    # Shuffle the list to randomize the dataset
    random.shuffle(image_files)

    # Calculate the number of files for each split
    total_files = len(image_files)
    train_size = int(train_ratio * total_files)
    test_size = int(test_ratio * total_files)

    # Split the dataset
    train_files = image_files[:train_size]
    test_files = image_files[train_size:train_size + test_size]
    valid_files = image_files[train_size + test_size:]

    # Move images and labels to respective subfolders
    def move_files(files, source_folder, dest_images_folder, dest_labels_folder):
        for file in files:
            # Move images
            shutil.copy(os.path.join(source_folder, file), os.path.join(dest_images_folder, file))
            
            # Move labels
            label_file = file.replace(".jpg", ".txt")
            shutil.copy(os.path.join(labels_folder, label_file), os.path.join(dest_labels_folder, label_file))

    move_files(train_files, images_folder, train_images_folder, train_labels_folder)
    move_files(test_files, images_folder, test_images_folder, test_labels_folder)
    move_files(valid_files, images_folder, valid_images_folder, valid_labels_folder)

def generate_yaml_from_json(json_file_path, yaml_file_path):
    def count_ids(data):
        id_count = 0
        for item in data.get('categories', []):
            if 'id' in item and isinstance(item['id'], int):
                id_count = max(id_count, item['id'] + 1)
        return id_count

    def extract_category_names(data):
        category_names = [category.get('name', '') for category in data.get('categories', [])]
        return category_names

    # Read JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Extract information from JSON
    nc = count_ids(data)
    names = extract_category_names(data)

    # Create data dictionary
    data_dict = {
        'train': '../train/images',
        'val': '../valid/images',
        'test': '../test/images',
        'nc': nc,
        'names': names
    }

    # Write data to YAML file
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(data_dict, yaml_file, default_flow_style=False)

    print(f"{yaml_file_path} file created successfully.")


def split_df():
    current_folder = os.getcwd()
    images_folder = os.path.join(current_folder, 'datasets_temp', 'images')
    labels_folder = os.path.join(current_folder, 'datasets_temp', 'labels')
    split_dataset(images_folder, labels_folder)

    json_file_path = os.path.join(current_folder, 'datasets_temp', 'notes.json')
    yaml_file_path = os.path.join(current_folder, 'datasets', 'data.yaml')
    generate_yaml_from_json(json_file_path, yaml_file_path)

