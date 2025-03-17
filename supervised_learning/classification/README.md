![Holberton School Logo](https://cdn.prod.website-files.com/6105315644a26f77912a1ada/63eea844ae4e3022154e2878_Holberton.png)

# Classification using Neural Networks

This project focuses on building a classification model from scratch using the MNIST dataset, which consists of handwritten digits (0-9). The goal is to progressively build more complex models: starting from a simple single neuron, moving to a neural network, then a deep learning model, and incorporating techniques like one-hot encoding and model storage using `pickle`.  

## Project Structure  

### 1. **Single Neuron (Binary Classification)**  
The project begins by implementing a **single artificial neuron** for a simple **binary classification task** (e.g., classifying if a digit is a '5' or not). It covers:  
  - Initializing weights and bias  
  - Forward and backward propagation  
  - Loss function (e.g., binary cross-entropy)  
  - Training the neuron using gradient descent  

### 2. **Neural Network (Multi-class Classification)**  
Next, a simple **neural network** with a single hidden layer is created to handle **multi-class classification**. The network will be able to classify all digits (0-9). Key steps include:  
  - Implementing the forward and backward propagation for a multi-layer network  
  - Using the softmax activation function for multi-class classification  
  - Using categorical cross-entropy as the loss function  

### 3. **Deep Neural Network (Deep Learning)**  
The neural network is extended to a **deep neural network** with multiple hidden layers. This will help the model learn more complex patterns in the data, enabling better performance on the MNIST dataset. Key aspects include:  
  - Adding more hidden layers  
  - Using activation functions like ReLU  
  - Optimizing the model with advanced gradient descent techniques (e.g., Adam)  

### 4. **One-hot Encoding and Storing Models with Pickle**  
In this step, **one-hot encoding** is applied to the labels for multi-class classification, and the trained models are **saved using `pickle`** for later use. Steps include:  
  - Implementing one-hot encoding for the MNIST labels  
  - Saving and loading trained models with `pickle`  

## Dataset  
The project uses the **MNIST dataset** for training and testing the models. It consists of 28x28 pixel images of handwritten digits, with a training set of 60,000 images and a test set of 10,000 images.

