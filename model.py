import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('C:/Users/shaik/Downloads/finalSentimentdata2.csv')  # Replace 'your_dataset.csv' with the actual file path

# Data Preprocessing
le = LabelEncoder()
df['sentiment'] = le.fit_transform(df['sentiment'])  # Convert sentiments to numerical labels

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['sentiment']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump((classifier, le, vectorizer), 'sentiment_model.pkl')

# Save evaluation results to a report
report_dict = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Classification Report': classification_report(y_test, y_pred, output_dict=True)
}

# Convert report to DataFrame for easier saving to a file
report_df = pd.DataFrame.from_dict(report_dict)
report_df.to_csv('sentiment_model_evaluation_report.csv', index=False)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Recommendations
print("Recommendations:")
print("- Consider exploring more advanced models for sentiment analysis.")
print("- Experiment with hyperparameter tuning to improve model performance.")
print("- Explore additional features or embeddings for better representation.")

# Future Work
print("\nFuture Work:")
print("- Fine-tune the model with a larger dataset for improved generalization.")
print("- Investigate the use of deep learning models for sentiment analysis.")
