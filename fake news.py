# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Function to clean text data
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Load the dataset
try:
    data = pd.read_csv('fake_news_data.csv')
except FileNotFoundError:
    print("Error: The file 'fake_news_data.csv' was not found.")
    exit()

# Ensure the dataset is loaded correctly
print("Dataset Head:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# Clean the text data
data['text'] = data['text'].apply(preprocess_text)

# Separate the text and labels
X = data['text']
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data with tuned parameters
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB(alpha=0.5)
classifier.fit(X_train_vec, y_train)

# Perform cross-validation
cv_scores = cross_val_score(classifier, vectorizer.transform(X), y, cv=5)
print(f"\nCross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Score: {cv_scores.mean()}")

# Make predictions on the test set
y_pred = classifier.predict(X_test_vec)

# Evaluate the model
print("\nAccuracy on Test Data:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize the Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Make predictions on new data
new_data = ['This is a real news article.', 'This is a fake news article.']
new_data_clean = [preprocess_text(text) for text in new_data]
new_vec = vectorizer.transform(new_data_clean)
predictions = classifier.predict(new_vec)

# Print predictions
print("\nPredictions on New Data:")
for text, prediction in zip(new_data, predictions):
    print(f"Article: {text}")
    print(f"Prediction: {'Real' if prediction == 'real' else 'Fake'}\n")

