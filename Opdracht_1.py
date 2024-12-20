import numpy as np
import matplotlib.pyplot as plt
import os
import gzip
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib



# Helper function to download files
def download_file(url, path):
    if not os.path.exists(path):
        print(f"Downloading {url} to {path}")
        urlretrieve(url, path)
    else:
        print(f"File {path} already exists.")

# URLs for the Fashion-MNIST dataset
base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
files = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# Download the dataset
data_dir = "./fashion-mnist-data"
os.makedirs(data_dir, exist_ok=True)

for key, filename in files.items():
    download_file(base_url + filename, os.path.join(data_dir, filename))

# Helper function to load Fashion-MNIST data
def load_data(images_path, labels_path):
    with gzip.open(images_path, "rb") as img_path, gzip.open(labels_path, "rb") as lbl_path:
        images = np.frombuffer(img_path.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        labels = np.frombuffer(lbl_path.read(), np.uint8, offset=8)
    return images, labels

# Load data
train_images, train_labels = load_data(
    os.path.join(data_dir, files["train_images"]),
    os.path.join(data_dir, files["train_labels"]),
)
test_images, test_labels = load_data(
    os.path.join(data_dir, files["test_images"]),
    os.path.join(data_dir, files["test_labels"]),
)

# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Flatten images for use in an MLP
X_train_flat = train_images.reshape(len(train_images), -1)
X_test_flat = test_images.reshape(len(test_images), -1)

# Split data into train/validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_flat, train_labels, test_size=0.2, random_state=42)

# Visualize some data
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

fig, ax = plt.subplots(3, 3, figsize=(8, 8))
for i, axi in enumerate(ax.flat):
    axi.imshow(train_images[i], cmap="gray")
    axi.set_title(class_names[train_labels[i]])
    axi.axis("off")
plt.tight_layout()
plt.show()

# Train an MLP Classifier
print("Training the MLP model...")
mlp = MLPClassifier(hidden_layer_sizes=(128,), activation="relu", solver="adam", max_iter=20, verbose=True)
mlp.fit(X_train, y_train)

# Evaluate the model
val_accuracy = mlp.score(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Predict on test data
y_pred = mlp.predict(X_test_flat)

# Confusion matrix
cm = confusion_matrix(test_labels, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="viridis", xticks_rotation=45)
plt.show()

# Visualize predictions
fig, ax = plt.subplots(3, 3, figsize=(8, 8))
for i, axi in enumerate(ax.flat):
    idx = np.random.randint(0, len(test_images))
    axi.imshow(test_images[idx], cmap="gray")
    true_label = class_names[test_labels[idx]]
    predicted_label = class_names[y_pred[idx]]
    axi.set_title(f"{true_label} ({predicted_label})" if true_label != predicted_label else true_label)
    axi.axis("off")
plt.tight_layout()
plt.show()


# Save the trained model to a file
model_path = "fashion_mnist_mlp.pkl"
joblib.dump(mlp, model_path)
print(f"Model saved to {model_path}")
