import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/disease_model.keras")

# Load scaler
scaler = joblib.load("models/scaler.pkl")

# Load label encoder
label_encoder = joblib.load("models/label_encoder.pkl")

# Patient symptoms
# Order:
# Fever, Cough, Headache, Fatigue,
# Body_Pain, Breathing_Difficulty,
# Sore_Throat, Nausea

print("\nEnter patient symptoms")
print("Enter 1 = Yes, 0 = No\n")

fever = int(input("Fever: "))
cough = int(input("Cough: "))
headache = int(input("Headache: "))
fatigue = int(input("Fatigue: "))
body_pain = int(input("Body Pain: "))
breathing = int(input("Breathing Difficulty: "))
sore_throat = int(input("Sore Throat: "))
nausea = int(input("Nausea: "))

patient = np.array([[
    fever,
    cough,
    headache,
    fatigue,
    body_pain,
    breathing,
    sore_throat,
    nausea
]])

# Scale patient data
patient_scaled = scaler.transform(patient)

# Predict
prediction = model.predict(patient_scaled, verbose=0)

# Get predicted disease
predicted_class = np.argmax(prediction, axis=1)[0]

disease = label_encoder.inverse_transform([predicted_class])[0]

# Confidence
confidence = np.max(prediction) * 100

print("\n==============================")
print("     DISEASE PREDICTION")
print("==============================")
print("Predicted Disease:", disease)
print("Confidence: {:.2f}%".format(confidence))