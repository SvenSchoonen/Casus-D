import os
import shutil
from PIL import Image

# Paths to the directories and the file containing bounding box data
file_path_complete = "/homes/sschoonen/Desktop/Casus_D/all_data/complete_images/"
file_path_idc = "/homes/sschoonen/Desktop/Casus_D/all_data/idc_regular/"
tekst = "/homes/sschoonen/Desktop/Casus_D/all_data/coords-idc.txt"

# Define the output directory structure
dataset_dir = "/homes/sschoonen/Desktop/Casus_D/dataset/"
images_dir = os.path.join(dataset_dir, "images")
labels_dir = os.path.join(dataset_dir, "labels")

# Function to calculate the bounding box data and normalize it
def calc_data_correct(data, w, h):
    name = data[0]
    x_min = int(data[1])
    y_min = int(data[2])
    x_max = int(data[3])
    y_max = int(data[4])
    
    # Calculate center and size in pixels
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    # Normalize the coordinates
    normalized_x_center = x_center / w
    normalized_y_center = y_center / h
    normalized_width = width / w
    normalized_height = height / h
    
    return [1, normalized_x_center, normalized_y_center, normalized_width, normalized_height]

# Function to get image dimensions (width and height)
def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size  # returns (width, height)

# Function to open and parse the coordinates data from the file
def open_data(path):
    lijst = []
    with open(path) as data:
        for i in data:
            data = i.replace("\n", "")
            data = data.split(",")
            lijst.append(data)
    return lijst

# Function to create a .txt file for each image with its bounding box information
def create_label_file(label_path, content):
    try:
        with open(label_path, 'w') as file:
            file.write(" ".join(map(str, content)) + "\n")
        print(f"Label file created: {label_path}")  # Debugging message
    except Exception as e:
        print(f"Error creating label file {label_path}: {e}")

# Function to copy image to the corresponding folder
def copy_image_to_folder(image_name, source_folder, dest_folder):
    try:
        image_path = os.path.join(source_folder, image_name + ".jpeg")
        if os.path.exists(image_path):
            dest_path = os.path.join(dest_folder, image_name + ".jpeg")
            shutil.copy(image_path, dest_path)
            print(f"Image copied to: {dest_path}")  # Debugging message
        else:
            print(f"Image not found: {image_name}.jpeg")
    except Exception as e:
        print(f"Error copying image {image_name}: {e}")

# Function to create the directories if they don't exist
def create_directories():
    # Make sure the directories exist
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    
    # Subdirectories for 'train' and 'val'
    for subdir in ['train', 'val']:
        train_subdir = os.path.join(images_dir, subdir)
        val_subdir = os.path.join(labels_dir, subdir)
        
        if not os.path.exists(train_subdir):
            os.makedirs(train_subdir)
            print(f"Created directory: {train_subdir}")
        if not os.path.exists(val_subdir):
            os.makedirs(val_subdir)
            print(f"Created directory: {val_subdir}")

# Function to distribute images and labels between 'train' and 'val' directories
def distribute_data(image_files, label_data, train_ratio=0.8):
    combined = list(zip(image_files, label_data))
    total_images = len(combined)
    train_count = int(total_images * train_ratio)
    
    train_set = combined[:train_count]
    val_set = combined[train_count:]
    
    return train_set, val_set

# Function to process images and labels
def process_images_and_labels():
    # Open the coordinates file
    data = open_data(tekst)
    print(f"Data from {tekst} loaded. Total entries: {len(data)}")  # Debugging message

    # Get image files and their respective bounding box data
    image_files = [d[0] for d in data]
    label_data = []
    
    # Track already processed images to avoid duplicates
    processed_images = set()  # To keep track of processed images
    for i, d in enumerate(data):
        image_name = d[0]
        
        # Skip if the image has already been processed
        if image_name in processed_images:
            continue
        
        image_path = os.path.join(file_path_complete, image_name + ".jpeg")
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found for {image_name}.jpeg. Skipping.")
            continue
        
        w, h = get_image_dimensions(image_path)
        label_data.append(calc_data_correct(d, w, h))
        processed_images.add(image_name)
    

    # Print some example labels to check if data is being generated
    print(f"Sample labels for first 5 images:")
    for label in label_data[:5]:
        print(label)

    # Split the data into training and validation sets
    train_set, val_set = distribute_data(image_files, label_data)
    # Create the directory structure
    create_directories()

    print("Starting to create label files and copy images...")

    for image_name, label in train_set:
        label_path = os.path.join(labels_dir, "train", image_name + ".txt")
        
        if not os.path.exists(label_path):
            create_label_file(label_path, label)

        copy_image_to_folder(image_name, file_path_complete, os.path.join(images_dir, "train"))
    
    for image_name, label in val_set:
        label_path = os.path.join(labels_dir, "val", image_name + ".txt")
        
        # Only create label file if it doesn't already exist
        if not os.path.exists(label_path):
            create_label_file(label_path, label)

        # Copy image to the val folder
        copy_image_to_folder(image_name, file_path_complete, os.path.join(images_dir, "val"))

# Main function to run the entire process
if __name__ == "__main__":
    process_images_and_labels()
