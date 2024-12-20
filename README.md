# Image Classification for Tissue Analysis

This repository contains a script to analyze tissue images (healthy vs unhealthy) using machine learning models, specifically **Random Forest** and **Multilayer Perceptron (MLP)** classifiers. The images are processed, filtered for certain conditions (e.g., high percentage of black pixels), and then classified based on tissue health.

## Features
- Load and filter images from a directory.
- Handle image resizing and preprocessing (grayscale, resize, etc.).
- Train machine learning models on a subset of the data.
- Evaluate models using accuracy and classification reports.
- Handle missing or low-quality images through filters.
- Visualize the percentage of unhealthy tissue per patient.

## Table of Contents

- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Model Details](#model-details)
- [Troubleshooting](#troubleshooting)
- [License](#license)
---

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/tissue-image-classification.git
    cd tissue-image-classification
    ```

2. **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install required packages:**

    You can install the required dependencies using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file should contain the following libraries:

    ```
    numpy
    opencv-python
    scikit-learn
    matplotlib
    random
    yaml
    ```

4. **Ensure you have a dataset:**

    - Prepare the image dataset of tissue scans (e.g., healthy and unhealthy tissue).
    - The directory structure should follow this format:
---

## Requirements

- Python 3.7+.
- Libraries: `numpy`, `opencv-python`, `scikit-learn`, `matplotlib`, `random`, `yaml`.
- Dataset: The image dataset should be structured as mentioned above, with images stored in subdirectories labeled `0` (healthy) and `1` (unhealthy).

---

## Usage

### 1. **Configure the script settings:**

Before running the script, make sure to configure the following parameters in the script:

- **`image_dir`**: The path to the directory containing your image dataset.
- **`target_size`**: The desired size for resizing the images (default is `(256, 256)`).
- **`sample_percentage`**: The percentage of images to sample from the dataset for training (default is `0.1` or 10%).
- **`test_size`**: The proportion of data to use for testing (default is `0.2` or 20%).
- **`new_image_path`**: The path to a new image for classification after training (optional).

#### Sample Configuration Example:

```python
image_dir = "images/"  # Path to your dataset
target_size = (256, 256)  # Resize all images to 256x256
sample_percentage = 0.1  # Use 10% of the images for training
test_size = 0.2  # 20% of the data will be used for testing
