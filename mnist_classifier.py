import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns



# Load MNIST dataset

print("Loading MNIST dataset...")

mnist = fetch_openml('mnist_784', version=1, parser='auto')

X, y = mnist.data, mnist.target



# Convert to numpy arrays and ensure correct types

X = np.array(X, dtype=float)

y = np.array(y, dtype=int)



# Use a subset for faster training (optional: remove for full dataset)

X = X[:10000]

y = y[:10000]



print(f"Dataset shape: {X.shape}")

print(f"Labels shape: {y.shape}")



# Split into train and test sets

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42

)



# Scale the features

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)



# Train a Logistic Regression classifier

print("\nTraining Logistic Regression model...")

model = LogisticRegression(max_iter=100, random_state=42, verbose=1)

model.fit(X_train_scaled, y_train)



# Make predictions

y_pred = model.predict(X_test_scaled)



# Evaluate the model

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")

print(classification_report(y_test, y_pred))



# Visualize some predictions

fig, axes = plt.subplots(2, 5, figsize=(12, 6))

axes = axes.ravel()



for i in range(10):

    axes[i].imshow(X_test[i].reshape(28, 28), cmap='gray')

    axes[i].set_title(f"True: {y_test[i]}\nPred: {y_pred[i]}")

    axes[i].axis('off')



plt.tight_layout()

plt.savefig('predictions.png', dpi=150, bbox_inches='tight')

print("\nSample predictions saved as 'predictions.png'")



# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title('Confusion Matrix')

plt.ylabel('True Label')

plt.xlabel('Predicted Label')

plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')

print("Confusion matrix saved as 'confusion_matrix.png'")



plt.show()

