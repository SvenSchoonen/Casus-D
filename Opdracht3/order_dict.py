import os
import shutil
import random

file_path_complete = "/homes/sschoonen/Desktop/Casus_D/images/10253"
file_path_idc = "/homes/sschoonen/Desktop/Casus_D/all_data/idc_regular/"
dataset_dir = "/homes/sschoonen/Desktop/Casus_D/Opdracht3/dataset/"

# 80% training, 20% validation)
train_ratio = 0.8
val_ratio = 0.2

# Create the directories if they don't exist
def create_directories():
    os.makedirs(os.path.join(dataset_dir, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'labels/val'), exist_ok=True)

# List all images and corresponding label files
def get_image_label_pairs():
    image_files = [f for f in os.listdir(file_path_complete) if f.endswith('.png')]
    label_files = [f.replace('.png', '.txt') for f in image_files]
    
    # Ensure that for each image, there's a corresponding label file
    image_label_pairs = [(image, label) for image, label in zip(image_files, label_files) 
                         if os.path.exists(os.path.join(file_path_idc, label))]
    
    return image_label_pairs

# Shuffle and split the data into training and validation
def split_data(image_label_pairs):
    random.shuffle(image_label_pairs)
    
    train_count = int(len(image_label_pairs) * train_ratio)
    train_data = image_label_pairs[:train_count]
    val_data = image_label_pairs[train_count:]
    
    return train_data, val_data

# Move files to their corresponding directories
def move_files(train_data, val_data):
    # Move training images and labels
    for image, label in train_data:
        shutil.move(os.path.join(file_path_complete, image), 
                    os.path.join(dataset_dir, 'images/train', image))
        shutil.move(os.path.join(file_path_idc, label), 
                    os.path.join(dataset_dir, 'labels/train', label))
    
    # Move valid images and labels
    for image, label in val_data:
        shutil.move(os.path.join(file_path_complete, image), 
                    os.path.join(dataset_dir, 'images/val', image))
        shutil.move(os.path.join(file_path_idc, label), 
                    os.path.join(dataset_dir, 'labels/val', label))

def main():
    create_directories()
    image_label_pairs = get_image_label_pairs()
    train_data, val_data = split_data(image_label_pairs)
    move_files(train_data, val_data)
    print(f"Training and validation data have been successfully organized.")

# Run the script
main()
