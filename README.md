# Emotion Detection with CNNs

## Introduction

This project aims to detect emotions from facial expressions using Convolutional Neural Networks (CNNs). The model is trained on the FER-2013 Facial Expression dataset from Kaggle, which consists of grayscale images of faces categorized into seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Dataset

The FER-2013 dataset contains 48x48 pixel grayscale images of faces. The dataset is split into a training set with 28,709 examples and a public test set with 3,589 examples.

**Dataset Link:** [FER-2013 Facial Expression Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

## Main Model Files

### 1. `TrainEmotionDetection.py`

This script is used to train the emotion detection model. Here's what it does:

- Utilizes Keras, an open-source neural network library in Python, for building and training the CNN model.
- Creates data generators for both training and testing purposes to read and preprocess the image data.
- Builds the CNN model using the Sequential model from Keras, consisting of convolutional layers, pooling layers, and a flatten layer.
- Uses categorical cross-entropy loss for optimization and trains the model.

The training process typically takes approximately 4 hours and 15 minutes.


### 2. `TestEmotionDetection.py`

This script is used to test the trained model in real-life scenarios, either using video files or laptop camera. Here's what it does:

- Utilizes OpenCV (cv2) to access live camera feed or read video content.
- Employs Haarcascade model for face detection before applying the emotion detection model to detect emotions.


## Applications
CNN-based emotion detection finds applications in diverse domains:

- **Facial Expression Analysis:** CNNs accurately detect emotions from images or videos, aiding mental health monitoring and personalized user experiences.
- **Human-Computer Interaction:** Systems adapt based on user emotions, enhancing empathy in virtual assistants and user interfaces.


Emotion detection with CNNs offers unparalleled potential in understanding human emotions and enhancing user experiences across various domains.
