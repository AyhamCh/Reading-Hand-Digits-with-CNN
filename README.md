# MNIST Classification using TensorFlow

This project demonstrates how to build and train a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset using **TensorFlow** and **TensorFlow Datasets**.

## Features
- Data preprocessing, including scaling and batching.
- CNN architecture with two convolutional layers and max-pooling.
- Early stopping to optimize training.
- Model evaluation and visualization of predictions.

## Dataset
The **MNIST** dataset consists of 70,000 grayscale images of handwritten digits (28x28 pixels). It is split into:
- **Training**: 60,000 images.
- **Testing**: 10,000 images.

The dataset is automatically downloaded using TensorFlow Datasets.

## Requirements
- Python 3.x
- TensorFlow
- TensorFlow Datasets
- Matplotlib
- NumPy

Install dependencies using:
```bash
pip install tensorflow tensorflow-datasets matplotlib numpy
```

## Usage

### 1. Data Preprocessing
- Scale pixel values to the range `[0, 1]`.
- Shuffle and split the dataset into training, validation, and testing sets.
- Batch the data for efficient processing.

### 2. Model Architecture
The CNN consists of:
1. **Conv2D**: 50 filters of size 5x5 with ReLU activation.
2. **MaxPooling2D**: Pool size of 2x2.
3. **Conv2D**: 50 filters of size 3x3 with ReLU activation.
4. **MaxPooling2D**: Pool size of 2x2.
5. **Flatten**: Converts 2D features to 1D.
6. **Dense**: 10 neurons (one for each class).

### 3. Training
- Loss: Sparse Categorical Crossentropy.
- Optimizer: Adam.
- Metric: Accuracy.
- Early stopping with patience of 2 epochs.

### 4. Evaluation
- Achieved **98.94% test accuracy**.
- Visualized predictions and probabilities for a sample image.

## Run the Code
Clone this repository and execute the script:
```bash
python mnist_classification.py
```

## Visualization
The script includes code to:
- Display a test image.
- Show prediction probabilities using a bar chart.

### Example Output
- Test loss: 0.0317
- Test accuracy: 98.94%
- Prediction probabilities for digit classes are visualized.

## Future Enhancements
- Experiment with more complex architectures.
- Add data augmentation.
- Test on other datasets like Fashion MNIST or CIFAR-10.

