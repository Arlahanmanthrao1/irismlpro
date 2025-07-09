import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Define column names for iris.data
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Load iris.data file
try:
    df = pd.read_csv('iris.data', header=None, names=columns)
except FileNotFoundError:
    print("❌ 'iris.data' file not found. Make sure it is in the root directory.")
    exit(1)

# Drop rows with missing data (if any)
df.dropna(inplace=True)

# Split features and target
X = df.drop('species', axis=1)
y = df['species']

# Encode species labels (e.g., Iris-setosa → 0, etc.)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train KNN classifier
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Save model and encoder
joblib.dump(knn_model, 'knn_iris_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("✅ KNN model and label encoder saved successfully.")
