from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Sample input for prediction (sepal length, sepal width, petal length, petal width)
sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(sample_input)
predicted_class = target_names[prediction[0]]

print(f"Input: {sample_input}")
print(f"Predicted Class: {predicted_class}")

