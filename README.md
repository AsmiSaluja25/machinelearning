# CIFAR-10 Object Recognition using SVM and CNN

## Project Overview
This project implements two machine learning approaches to classify images from the CIFAR-10 dataset into 10 object categories, including airplanes, cars, animals, and ships.  

- **Approach 1:** Support Vector Machine (SVM) with Histogram of Oriented Gradients (HOG) feature extraction and Principal Component Analysis (PCA) for dimensionality reduction.  
- **Approach 2:** Convolutional Neural Network (CNN) to automatically learn hierarchical features from raw images.  

The goal is to evaluate and compare classical ML (SVM) and deep learning (CNN) approaches for image classification, providing insights into performance, strengths, and weaknesses.

---

## **Dataset**
- CIFAR-10: 60,000 images (32x32x3) in 10 classes.  
  - Training set: 50,000 images  
  - Testing set: 10,000 images  
- Data stored as 4D arrays `(height, width, channels, samples)`.

---

## **Implementation**

### 1. Preprocessing
- Transpose 4D arrays to `(samples, height, width, channels)` format.  
- Convert images to grayscale for HOG feature extraction.  
- Standardize features using `StandardScaler` for SVM.  

### 2. Feature Extraction
- **HOG Features:** Extracted edge and shape information from images.  
- **PCA:** Reduced dimensionality to 300 components to improve SVM training efficiency.  

### 3. Model Training

#### SVM Classifier
- Trained on standardized HOG features.  
- Achieved **47.3% accuracy** on the test set.  
- Strengths: fast training, interpretable features, good performance on horses.  

#### CNN Classifier
- Normalized pixel values to [0,1].  
- Built a deep CNN with multiple Conv2D, BatchNorm, MaxPooling, Dropout layers.  
- Trained for 20 epochs using Adam optimizer and sparse categorical cross-entropy loss.  
- Achieved **55.3% accuracy** on the test set.  
- Strengths: learns features automatically, better overall performance, excels in ship and airplane classification.  

---

## **Evaluation**
- Confusion matrices and classification reports generated for both SVM and CNN.  
- Comparative analysis highlights CNN’s superiority but also its failure in truck classification.  
- Insights used for suggesting improvements such as data augmentation, longer training, and hyperparameter tuning.  

---

## **Performance Comparison**

| Model | Test Accuracy | Key Strengths |
|-------|---------------|---------------|
| SVM   | 47.3%         | Fast training, interpretable features, good on horses |
| CNN   | 55.3%         | Better overall performance, learns features automatically, strong on ships & airplanes |

---

## **Skills Learned**
- Image preprocessing and feature extraction (HOG)  
- Dimensionality reduction with PCA  
- Machine Learning: SVM, classification metrics, confusion matrices  
- Deep Learning: CNN architecture design, training, evaluation  
- Data visualization and exploratory data analysis  
- Comparative analysis of classical ML vs deep learning

---

## **Usage**
1. Clone this repository  
2. Install dependencies:  
   ```bash
   pip install numpy matplotlib seaborn scikit-learn scikit-image tensorflow pandas
