# **Pakistani Currency Classification Using CNN**

This repository contains a project for classifying Pakistani currency notes using a Convolutional Neural Network (CNN). The dataset consists of images of Pakistani currency notes in denominations of 10, 20, 50, 100, 500, 1000, and 5000 PKR. The model was built using TensorFlow and Keras, and evaluated with metrics such as accuracy and classification reports.

## Table of Contents
- Dataset
- Model Architecture
- Data Preprocessing and Augmentation
- Training
- Evaluation
- Results

## Dataset
The dataset used in this project contains images of Pakistani currency notes, organized into seven classes representing different denominations. The images were split into training, validation, and test sets with a ratio of 70:15:15.

- **Train images**: 4,198
- **Validation images**: 665
- **Test images**: 677
- **Classes**: 7 (10, 20, 50, 100, 500, 1000, 5000 PKR)

The dataset was stored in Google Drive and accessed via Google Colab for processing.

## Model Architecture
The CNN architecture was built using Keras' Sequential API. The network includes:

- **Input layer**: Accepts images of shape (64, 64, 3).
- **Convolutional layers**: Two convolutional layers with 32 filters each, kernel size of 3, and ReLU activation, followed by max-pooling layers with a pool size of 2.
- **Flatten layer**: Converts the 2D feature maps into a 1D vector.
- **Dense layers**: A fully connected layer with 128 units and ReLU activation, followed by an output layer with 7 units (one per class) and softmax activation.

## Summary of the model:
- **Input shape**: (64, 64, 3)
- **Output**: 7 classes (softmax)

## Data Preprocessing and Augmentation
The images were preprocessed and augmented to improve model generalization:

- **Rescaling**: Pixel values were normalized to the range [0, 1] by dividing by 255.
- **Data augmentation** (applied to training set):
  - Shear range: 0.2
  - Zoom range: 0.2
  - Horizontal flip: Enabled
- **Data generators**: TensorFlow's `ImageDataGenerator` was used to load and augment images in batches of 32, resizing them to (64, 64).

The dataset was split into training (70%), validation (15%), and test (15%) sets, with folder structures created for each split.

## Training
The model was trained with the following configurations:

- **Optimizer**: Adam
- **Loss function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 25 (trained twice, totaling 50 epochs)
- **Batch size**: 32

No callbacks such as EarlyStopping or ModelCheckpoint were used in the provided code.

## Evaluation
The model was evaluated on the test set using:

- **Accuracy**: Computed during training and validation.
- **Classification report**: Generated using `sklearn.metrics.classification_report`, providing precision, recall, and F1-score for each class.

## Key metrics:
- **Test accuracy**: Approximately 95%
- **Classification report**:
  - Precision, recall, and F1-score ranged from 0.93 to 0.99 across classes.
  - Macro average F1-score: 0.95
  - Weighted average F1-score: 0.95

The detailed classification report is available in the notebook's output.

## Results
The CNN model achieved high accuracy (95%) on the test set, demonstrating effective classification of Pakistani currency notes. The classification report indicates balanced performance across all denominations, with minor variations in precision and recall for some classes (e.g., class '10' had a slightly lower recall of 0.86). The training and validation accuracy improved steadily over epochs, with the model showing no significant signs of overfitting in the provided output.