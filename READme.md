# MNIST Digit Classifier

A machine learning project that trains a Logistic Regression classifier on the MNIST handwritten digits dataset.

## GitHub Description

MNIST Handwritten Digit Classifier using Logistic Regression. This machine learning project implements a complete pipeline for digit recognition on the classic MNIST dataset. The project loads 10,000 handwritten digit samples, preprocesses the data using StandardScaler, and trains a Logistic Regression classifier to predict digits from 0-9. The implementation includes comprehensive model evaluation with accuracy metrics, classification reports, and visualizations. The code generates prediction visualizations showing true vs predicted labels and a confusion matrix heatmap. Built with scikit-learn, numpy, matplotlib, and seaborn. Perfect for learning machine learning fundamentals and image classification techniques.

## Features

- Loads MNIST dataset (10,000 samples)
- Trains a Logistic Regression model
- Evaluates model performance
- Visualizes predictions
- Generates confusion matrix

## Installation

Install required packages:

```bash
pip install numpy matplotlib scikit-learn seaborn
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

Run the script:

```bash
python mnist_classifier.py
```

## Output

The script will:
1. Load and preprocess the MNIST dataset
2. Train a Logistic Regression classifier
3. Display accuracy and classification report
4. Save visualization images:
   - `predictions.png` - Sample predictions
   - `confusion_matrix.png` - Confusion matrix heatmap

## Requirements

- Python 3.x
- numpy
- matplotlib
- scikit-learn
- seaborn

