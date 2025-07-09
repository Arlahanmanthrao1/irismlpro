import joblib
import numpy as np
from sklearn.datasets import load_iris

# Load the saved KNN model
knn = joblib.load('knn_iris_model.pkl')

# Load Iris target names
iris = load_iris()
target_names = iris.target_names

# Sample input: [sepal length, sepal width, petal length, petal width]
sample_input = np.array([[5.9, 3.0, 5.1, 1.8]])

# Make prediction
prediction = knn.predict(sample_input)
predicted_class = target_names[prediction[0]]

print(f"Input: {sample_input}")
print(f"Predicted Class: {predicted_class}")
