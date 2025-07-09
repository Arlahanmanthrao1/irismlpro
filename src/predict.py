import joblib
import numpy as np

# Load the trained model and label encoder
knn_model = joblib.load('knn_iris_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Sample input for prediction: [sepal length, sepal width, petal length, petal width]
sample_input = np.array([[6.0, 3.0, 4.8, 1.8]])  # You can modify this

# Predict the class (numerical)
predicted_class_num = knn_model.predict(sample_input)

# Decode the numerical label to original species name
predicted_species = label_encoder.inverse_transform(predicted_class_num)

# Output
print("📥 Input Data:", sample_input[0])
print("🔍 Predicted Species:", predicted_species[0])
