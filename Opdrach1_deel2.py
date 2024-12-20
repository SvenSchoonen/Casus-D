from PIL import Image
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Load the trained model
mlp = joblib.load("fashion_mnist_mlp.pkl")

# Class names 
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Shapping image
def preprocess_image(image_path):
    img = Image.open(image_path).convert("L").resize((28, 28))
    return np.array(img).reshape(1, -1) / 255.0

# Predict and display results with own picture


# Make path relative

# Construct the relative path
image_path = r"C:\Users\Sven Schoonen\Desktop\sCHOOL\Casus_D\takeout-001\foto.jpg"
processed_image = preprocess_image(image_path)

predicted_class = mlp.predict(processed_image)[0]
predicted_probabilities = mlp.predict_proba(processed_image)[0]

print("Prediction:", class_names[predicted_class])
print("Probabilities:")
for cls, prob in zip(class_names, predicted_probabilities):
    print(f"{cls}: {prob:.2%}")

# Show the image
plt.imshow(Image.open(image_path), cmap="gray")
plt.title(f"Prediction: {class_names[predicted_class]}")
plt.axis("off")
plt.show()

