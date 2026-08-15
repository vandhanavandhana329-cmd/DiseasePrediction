import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
print("Libraries imported successfully")


# Load dataset
df = pd.read_csv("dataset/diabetes.csv")
print("\nDataset loaded successfully!")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)


# Separate features and target
X = df.drop("Disease", axis=1)
y = df["Disease"]
print("\nFeatures:")
print(X.columns)
print("\nTarget diseases:")
print(y.unique())


# Encode disease names into numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print("\nEncoded disease classes:")
print(label_encoder.classes_)

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("\nFeature scaling completed successfully!")

# Create Neural Network
model = Sequential()
# Input layer + Hidden layer
model.add(Dense(16, input_dim=X_train.shape[1], activation="relu"))
# Second hidden layer
model.add(Dense(8, activation="relu"))
# Output layer - 5 diseases
model.add(Dense(5, activation="softmax"))
print("\nNeural Network model created successfully!")

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
print("Model compiled successfully!")


# Train the Neural Network
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)
print("\nModel training completed successfully!")

# Test the model
y_pred_prob = model.predict(X_test)
# Convert probabilities into predicted class
y_pred = np.argmax(y_pred_prob, axis=1)
print("\nPrediction completed!")

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted Disease")
plt.ylabel("Actual Disease")
plt.title("Disease Prediction - Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.show()


# Accuracy Graph
plt.figure(figsize=(7, 5))
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend(["Training Accuracy", "Validation Accuracy"])
plt.tight_layout()
plt.savefig("results/accuracy_graph.png")


# Save trained model
model.save("models/disease_model.keras")
# Save scaler and label encoder
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
print("Scaler saved successfully!")
print("Label encoder saved successfully!")
print("\nModel saved successfully!")