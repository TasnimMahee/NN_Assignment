# Pneumonia Detection using CNN 

## 📌 Overview  
This project involves building a Convolutional Neural Network (CNN) to classify pediatric chest X-ray images into two categories: **Pneumonia** and **Normal**. The goal is to develop an accurate model that can assist healthcare professionals by automating the identification of pneumonia from X-ray images.


## 📊 Dataset  
- The dataset consists of **5,863** anterior-posterior chest X-ray images collected from pediatric patients aged 1 to 5 years.  
- Images are organized into three folders: **train**, **validation (val)**, and **test**, each containing subfolders for the two classes: Pneumonia and Normal.


## ⚙️ Approach & Methodology  

1. **Data Loading & Preprocessing**  
   - Used TensorFlow’s `ImageDataGenerator` to load images from folders and perform real-time data augmentation (e.g., rescaling, flipping) to reduce overfitting.  
   - Images resized and normalized for feeding into the CNN.

2. **Model Architecture**  
   - Built a CNN with several convolutional and max-pooling layers to extract features from images.  
   - Added dropout layers to reduce overfitting.  
   - Used `ReLU` activations and a `sigmoid` activation in the output layer for binary classification.

3. **Training**  
   - Trained the model for 10 epochs using Adam optimizer and binary cross-entropy loss.  
   - Monitored training and validation accuracy and loss to evaluate learning progress.

4. **Evaluation**  
   - Tested the model on unseen test data and measured accuracy.


## 📈 Results  
- **Training Accuracy:** Reached up to ~95%.  
- **Validation Accuracy:** Varied between 75% and 87.5% during training, indicating some variance likely due to limited validation data.  
- **Test Accuracy:** Achieved 86.5% accuracy on the test dataset, demonstrating good predictive performance.

## ⚠️ Warnings & Notes  
- TensorFlow logs indicate that some optimizations and GPU support are not available on native Windows (recommend using WSL2 or TensorFlow-DirectML for GPU acceleration).  
- The model is saved in the legacy HDF5 `.h5` format; switching to the native `.keras` format is recommended for future use.  

## 🔍 Findings & Insights  
- The CNN successfully learned meaningful features from chest X-rays to classify pneumonia with reasonable accuracy.  
- The validation accuracy fluctuations suggest potential overfitting, which can be improved with further tuning, more data, or additional regularization techniques.  
- Automating pneumonia detection can assist healthcare providers in quicker diagnosis and treatment.


## 🧰 Tools & Libraries  
- Python 3.13  
- TensorFlow (Keras)  
- Matplotlib  
- PIL (for image processing)  
