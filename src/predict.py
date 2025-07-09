import joblib
import numpy as np

# Load model and label encoder
knn = joblib.load('knn_iris_model.pkl')
le = joblib.load('label_encoder.pkl')

# Example prediction input
sample = np.array([[5.9, 3.0, 5.1, 1.8]])

# Predict
prediction = knn.predict(sample)
predicted_species = le.inverse_transform(prediction)

print(f"Input: {sample}")
print(f"Predicted Species: {predicted_species[0]}")
