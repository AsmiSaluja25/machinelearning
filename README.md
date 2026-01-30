# machinelearning
CIFAR-10 Object Recognition using SVM and CNN

This project demonstrates two machine learning approaches for image classification on the CIFAR-10 dataset:

Support Vector Machine (SVM) using Histogram of Oriented Gradients (HOG) features and Principal Component Analysis (PCA)

Convolutional Neural Network (CNN) for end-to-end feature learning and classification

Table of Contents

Overview

Dataset

Methods

SVM with HOG and PCA

Convolutional Neural Network

Results

Performance Comparison

Future Improvements

Libraries

Overview

The goal of this project is to classify images from CIFAR-10 into 10 object categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
We implemented:

A traditional machine learning approach (SVM) using extracted HOG features.

A deep learning approach (CNN) for automatic feature extraction and classification.

This notebook provides a comparison of model performance, visualizations, and key insights.

Dataset

The CIFAR-10 dataset consists of 60,000 images (32×32 pixels, RGB) divided into:

50,000 training images

10,000 testing images

Each image belongs to one of 10 classes. Images are represented as 4D arrays: (height, width, channels, samples).

Methods
SVM with HOG and PCA

Convert images to grayscale and extract HOG features to capture shapes and edges.

Standardize features using StandardScaler (zero mean, unit variance).

Dimensionality reduction with PCA to reduce feature vectors to 300 components.

Train SVM classifier with RBF kernel (C=10, gamma=0.001).

Evaluate using accuracy, classification report, and confusion matrix.

from sklearn.svm import SVC
svm_classifier = SVC(kernel='rbf', C=10, gamma=0.001)
svm_classifier.fit(training_image_standardised, training_label)
y_pred = svm_classifier.predict(testing_image_standardised)
accuracy_score(testing_label, y_pred)


Strengths: Fast training, interpretable features, performs well on specific categories like horses.

Convolutional Neural Network

Normalize images to [0,1] range.

Define CNN architecture with multiple Conv2D, BatchNormalization, MaxPooling, and Dropout layers.

Compile model with Adam optimizer and sparse categorical crossentropy.

Train on training images for 20 epochs with a 10% validation split.

Evaluate on the test set using accuracy, classification report, and confusion matrix.

cnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = cnn_model.fit(training_image, training_label, batch_size=32, epochs=20, validation_split=0.1)


Strengths: Learns features automatically, better overall accuracy than SVM, excels on categories like ships and airplanes.

Results
Model	Test Accuracy	Key Strengths
SVM	47.3%	Fast training, interpretable features, good on horses
CNN	55.3%	Better overall performance, learns features automatically

Key Findings:

CNN outperforms SVM in overall classification.

Both models struggle with similar animal categories (bird, cat, dog).

CNN shows high precision on ships (83%) and airplanes (76%).

SVM performs better on horses (70%) than CNN (67%).

CNN fails on truck classification (0%), indicating a data or architecture issue.

Performance Visualization

Confusion Matrix for CNN

Confusion Matrix for SVM

Future Improvements

Train CNN for more epochs (50–100) with early stopping.

Implement data augmentation (flips, rotations, shifts).

Hyperparameter tuning for SVM and CNN.

Investigate truck classification failure in CNN.

Libraries

numpy, matplotlib, seaborn, pandas

scikit-learn (SVM, PCA, StandardScaler, metrics)

scikit-image (HOG feature extraction)

tensorflow, keras (CNN modeling)
