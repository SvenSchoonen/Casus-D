import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import yaml
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
import random
# add config path
image_idc = "idc/"  

target_size = (256, 256)  # Nieuwe grootte voor alle images


# Functie om geldige afbeeldingen te filteren op basis van de gewenste grootte en black pixel percentage
def filter_valid_images(image_dir, target_size=(256, 256), black_pixel_threshold=0.5):
    valid_images = []
    valid_labels = []
    patient_data = {}
    print("Start loading data")
    
    for patient_id in os.listdir(image_dir):
        patient_path = os.path.join(image_dir, patient_id)
        if os.path.isdir(patient_path):
            for label in [0, 1]:  # 0 for healthy, 1 for unhealthy
                label_dir = os.path.join(patient_path, str(label))
                if os.path.isdir(label_dir):
                    for filename in os.listdir(label_dir):
                        if filename.endswith(".png"):
                            img_path = os.path.join(label_dir, filename)
                            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if image is not None:
                                # Resize images if smaller than target size
                                if image.shape != target_size:
                                    image_resized = cv2.resize(image, target_size)
                                else:
                                    image_resized = image

                                # Calculate the percentage of black pixels in the image black == 0
                                black_pixels = np.sum(image_resized == 0)  
                                total_pixels = image_resized.size 
                                black_pixel_percentage = black_pixels / total_pixels # average

                                # Skip the image if it has more than the threshold percentage of black pixels
                                if black_pixel_percentage > black_pixel_threshold:
                                    print("Got to much black pixels:", patient_path)
                                    continue

                                valid_images.append(image_resized)
                                valid_labels.append(label)

                                # Add data per patient and category
                                if patient_id not in patient_data:
                                    patient_data[patient_id] = {"healthy": 0, "unhealthy": 0}
                                if label == 0:
                                    patient_data[patient_id]["healthy"] += 1
                                else:
                                    patient_data[patient_id]["unhealthy"] += 1
    print("Done")
    return np.array(valid_images), np.array(valid_labels), patient_data


# Laad de gegevens
print('Laad images')
images, labels, patient_data = filter_valid_images(image_dir, target_size=target_size)

missing_values = np.any(np.isnan(images))
print("Zijn er missende waarden in de dataset?", missing_values)

# Gezond en ongezond data
healthy_count = np.sum(labels == 0)
unhealthy_count = np.sum(labels == 1)
print("Aantal gezonde uitsneden:(pixels)", healthy_count)
print("Aantal ongezonde uitsneden:(pixels)", unhealthy_count)
ratio = healthy_count / unhealthy_count
print("Verhouding gezonde vs ongezonde uitsneden:(%)", ratio)

# 3. Percentage van de totale scan dat ongezond weefsel is
patient_unhealthy_percentages = {}
for patient_id, data in patient_data.items():
    total_slices = data["healthy"] + data["unhealthy"]
    unhealthy_percentage = (data["unhealthy"] / total_slices) * 100
    patient_unhealthy_percentages[patient_id] = unhealthy_percentage

average_unhealthy_percentage = np.mean(list(patient_unhealthy_percentages.values()))
print("Gemiddeld percentage ongezond weefsel per patiënt:", average_unhealthy_percentage)

# Plot percentage ongezond weefsel per patiënt
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(patient_unhealthy_percentages.keys(), patient_unhealthy_percentages.values())
ax.set_title('Percentage ongezond weefsel per patiënt')
ax.set_xlabel('Patiënt ID')
ax.set_ylabel('Percentage ongezond weefsel')
plt.xticks(rotation=90)
plt.show()

# Sampling 10% of the images
def sample_images(images, labels, sample_percentage=0.1):
    sample_size = int(len(images) * sample_percentage)  # 10% of the dataset
    indices = list(range(len(images)))
    sampled_indices = random.sample(indices, sample_size)  # Randomly sample indices
    sampled_images = images[sampled_indices]
    sampled_labels = labels[sampled_indices]
    return sampled_images, sampled_labels

# Sample 10% of the images
print("Sampling 10% of the images...")
sampled_images, sampled_labels = sample_images(images, labels, sample_percentage=0.1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(sampled_images, sampled_labels, test_size=0.2, random_state=42)

# Reshape the data for model input (flatten images into 1D arrays)
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

# Random Forest Model (for comparison)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_reshaped, y_train)

# Predict and evaluate Random Forest
y_pred_rf = rf_model.predict(X_test_reshaped)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# MLP Model (Neural Network)
mlp_model = MLPClassifier(hidden_layer_sizes=(128,), activation="relu", solver="adam", max_iter=20, random_state=42)
mlp_model.fit(X_train_reshaped, y_train)

# Predict and evaluate MLP
y_pred_mlp = mlp_model.predict(X_test_reshaped)
print("MLP Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("MLP Classification Report:")
print(classification_report(y_test, y_pred_mlp))

# Predict a new image
new_image_path = "/path/to/new/image.png"  # Update with the path to your new image
new_image = cv2.imread(new_image_path, cv2.IMREAD_GRAYSCALE)
new_image_resized = cv2.resize(new_image, target_size)  # Resize to the target size
new_image_reshaped = new_image_resized.reshape(1, -1)  # Flatten the image

# Predict for the new image with MLP
prediction = mlp_model.predict(new_image_reshaped)
print("Prediction for the new image (MLP):", "Healthy" if prediction[0] == 0 else "Unhealthy")
