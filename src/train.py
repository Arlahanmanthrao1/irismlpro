import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load iris.data
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv('iris.data', header=None, names=columns)

# Drop any rows with missing values (just in case)
df.dropna(inplace=True)

# Features and target
X = df.drop('species', axis=1)
y = df['species']

# Encode species labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Save model and label encoder
joblib.dump(knn, 'knn_iris_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("✅ KNN model trained and saved.")
