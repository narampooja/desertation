Disease Detection Using Deep Learning Models

This repository contains the implementation of a multi-class disease detection system using Convolutional Neural Networks (CNNs). The project includes classification of medical images from chest X-rays (pneumonia vs. normal) and melanoma images (benign vs. malignant). This guide will walk you through the process of running the code.

The project applies CNN architectures for early disease detection using medical imaging. It combines two datasets:
1. Chest X-rays (pneumonia vs. normal).
2. Melanoma images (benign vs. malignant).

The system is designed to:
Train and evaluate multiple CNN-based models.
Visualize training progress and predictions.
Generate Receiver Operating Characteristic (ROC) curves and Area Under the Curve (AUC) scores.

Features
Supports data preprocessing and augmentation.
Includes Grad-CAM visualization for interpretability.
Implements models such as CNN, VGG, ResNet, and EfficientNet.
Provides visualizations of predictions on test samples.

Installation
Prerequisites
Make sure the following are installed:
Python (>= 3.8)
TensorFlow (>= 2.10)
NumPy
Matplotlib
Pandas
scikit-learn

Installing Dependencies

Step 1: Download Datasets
Chest X-ray Dataset: Download from Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
Melanoma Dataset: Download from Kaggle: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images

Step 2: Organize the Data
Ensure the data is organized into the following structure:
data/
|-- chest_xray/
|   |-- train/
|   |-- test/
|   |-- val/
|-- melanoma/
|   |-- train/
|   |-- test/
|-- combined/

Step 3: Preprocess Images
Resize all images to 128x128 pixels.
Normalize pixel values to the range [0, 1].
The code includes preprocessing steps, so you just need to ensure the datasets are placed correctly.

How to Run the Code
Step 1: Clone the Repository
Clone this repository to your local machine:
git clone https://github.com/your-username/disease-detection.git
cd disease-detection

Step 2: Prepare the Environment
Ensure the datasets are organized as described in the Dataset Preparation section.

Step 3: Run the Training Script
To train the CNN model, execute the following command:
python train_combined_model.py
This script will:
Preprocess and augment the data.
Train the CNN model on the combined dataset.
Save the trained model as model.h5.

Step 4: Evaluate the Model
To evaluate the trained model and generate metrics:
python evaluate_model.py

Step 5: Visualize Predictions
To visualize predictions on test samples:
python visualize_predictions.py
This script will display random predictions with true labels (green for correct, red for incorrect).

Note
For any issues or queries, feel free to open an issue in this repository.
