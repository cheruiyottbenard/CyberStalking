'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import numpy as np
import tensorflowjs as tfjs

# Step 1: Load and preprocess the dataset
data = pd.read_csv('sexism.csv')  # Replace 'your_dataset.csv' with your dataset file

# Drop rows with missing values
data = data.dropna(subset=['Text', 'oh_label'])

# Extract text data and labels
X = data['Text']
y = data['oh_label']

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a Naive Bayes classifier
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_counts, y_train)

# Step 4: Evaluate the classifier
y_pred = nb_classifier.predict(X_test_counts)
f1_accuracy = f1_score(y_test, y_pred)

print("F1 Score:", f1_accuracy)

tfjs.converters.save_keras_model(nb_classifier, 'tfjs_model')

'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib
import tensorflowjs as tfjs

# Step 1: Load and preprocess the dataset
data = pd.read_csv('sexism.csv')

# Drop rows with missing values
data = data.dropna(subset=['Text', 'oh_label'])

# Extract text data and labels
X = data['Text']
y = data['oh_label']

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Vectorize the text data
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Step 4: Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_counts, y_train)

# Step 5: Evaluate the classifier
y_pred = rf_classifier.predict(X_test_counts)
f1_accuracy = f1_score(y_test, y_pred)
print("F1 Score:", f1_accuracy)

# Step 6: Save the trained Random Forest classifier
joblib.dump(rf_classifier, 'rf_model.joblib')

