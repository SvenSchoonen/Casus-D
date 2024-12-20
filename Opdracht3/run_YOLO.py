import os
import cv2
import matplotlib.pyplot as plt
import yaml
from ultralytics import YOLO

# Load config.yaml file
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_model(config):
    model = YOLO(config['yolov8_model'])
    model.train(
        data=config['dataset_yaml_path'],
        epochs=config['train_config']['epochs'],
        single_cls=config['train_config']['single_cls'],
        batch=config['train_config']['batch_size'],
        imgsz=config['train_config']['imgsz']
    )

def predict_on_new_data(config, image_path):
    # Load the trained model
    model = YOLO(config['output_model_path'] + 'weights/best.pt')  # Point to the correct path of your trained model
    
    # Predict the image
    results = model(image_path)  # Perform inference
    
    # Show the results
    results.show()  # This will open a default image viewer with the prediction results
    
    # Save the predictions to the output directory
    results.save()  # This saves the prediction result images to the directory
    
    # Convert results to a pandas DataFrame for easy inspection
    predictions_df = results.pandas().xywh
    print("Predictions (in xywh format):")
    print(predictions_df)
    
    # Visualize predictions by drawing boxes on the image
    image = cv2.imread(image_path)  # Read the input image
    
    # Loop through predictions and draw boxes on the image
    for *box, conf, cls in results.pred[0]:  # Loop through detected objects
        x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
        label = f"{model.names[int(cls)]} {conf:.2f}"  # Class name and confidence
        
        # Draw a rectangle around the predicted object
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Put the label text above the bounding box
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with bounding boxes
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display with matplotlib
    plt.axis('off')  # Turn off axis labels for better visualization
    plt.show()

if __name__ == "__main__":
    config = load_config()  # Load the config file

    # Uncomment to train the model (if not already trained)
    # train_model(config)

    # Make predictions on a new image (test image from config)
    predict_on_new_data(config, config['paths']['test_image'])
