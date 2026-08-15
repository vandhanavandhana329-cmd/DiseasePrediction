# Disease Prediction Using Neural Network

## Project Description

This project uses a Neural Network to predict diseases based on patient symptoms.

The model takes symptoms such as fever, cough, headache, fatigue, body pain, breathing difficulty, sore throat, and nausea as input and predicts the most likely disease.

## Diseases Predicted

- Common Cold
- Flu
- Food Poisoning
- Migraine
- Pneumonia

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn

## Dataset

The dataset contains 500 records with 8 symptom features and 1 target disease column.

### Input Features

- Fever
- Cough
- Headache
- Fatigue
- Body Pain
- Breathing Difficulty
- Sore Throat
- Nausea

## Machine Learning Workflow

1. Load the dataset
2. Data preprocessing
3. Label encoding
4. Train-test split
5. Feature scaling
6. Neural Network model creation
7. Model training
8. Model evaluation
9. Confusion matrix
10. Disease prediction

## Neural Network Architecture

- Input Layer: 8 features
- Hidden Layer 1: 16 neurons
- Hidden Layer 2: 8 neurons
- Output Layer: 5 classes
- Activation Function: ReLU and Softmax
- Optimizer: Adam

## Model Performance

The model achieved approximately 72% test accuracy on the current dataset.

## Project Structure

```text
Disease-Prediction-Neural-Network
│
├── dataset
│   └── diabetes.csv
│
├── models
│   ├── disease_model.keras
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── results
│   ├── accuracy_graph.png
│   └── confusion_matrix.png
│
├── src
│   ├── train_model.py
│   └── predict.py
│
└── README.md