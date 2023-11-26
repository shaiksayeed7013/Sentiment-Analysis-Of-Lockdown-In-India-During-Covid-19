import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Load an image for lockdown
lockdown_image = Image.open("lockdown_image.jpeg")  # Replace with the actual file path of your image

# Display the image in Streamlit
st.image(lockdown_image, caption="Lockdown Image", use_column_width=True)

# Load your dataset
df = pd.read_csv('finalSentimentdata2.csv')  # Replace 'your_dataset.csv' with the actual file path


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

# Save the trained model
joblib.dump((classifier, le, vectorizer), 'sentiment_model.pkl')

# Application to predict sentiment for a given keyword
def predict_sentiment(keyword):
    # Load the trained model
    model, label_encoder, text_vectorizer = joblib.load('sentiment_model.pkl')

    # Predict sentiment for the keyword
    keyword_vectorized = text_vectorizer.transform([keyword])
    prediction = model.predict(keyword_vectorized)
    predicted_sentiment = label_encoder.inverse_transform(prediction)

    return predicted_sentiment[0]

# Streamlit Application
st.title("Twitter Sentiment Analysis during Lockdown in India")

# User input for keyword
user_input = st.text_input("Enter a keyword:")

# Perform sentiment prediction when the user clicks the button
if st.button("Predict Sentiment"):
    result = predict_sentiment(user_input)
    
    # Map sentiments to emojis
    emoji_mapping = {
        "happy": "😄",
        "sad": "😢",
        "joy": "😊",
        "anger": "😡",
        "fear": "😨",
        # Add more mappings as needed
    }

    # Display sentiment with emoji
    if result.lower() in emoji_mapping:
        emoji = emoji_mapping[result.lower()]
        st.success(f"Sentiment for '{user_input}': {result} {emoji}")
    else:
        st.success(f"Sentiment for '{user_input}': {result}")

# Add a separate section for evaluation results and confusion matrix
if st.checkbox("Show Evaluation Results"):
    st.subheader("Evaluation Results")

    # Evaluate the classifier
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_report_str = classification_report(y_test, y_pred)

    # Display accuracy and classification report
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
    st.text("Classification Report:")
    st.code(classification_report_str)

if st.checkbox("Show Confusion Matrix"):
    st.subheader("Confusion Matrix")

    # Create a confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    st.pyplot(fig)

# Recommendations and Future Work
st.subheader("Recommendations and Future Work")
st.markdown("- Consider exploring more advanced models for sentiment analysis.")
st.markdown("- Experiment with hyperparameter tuning to improve model performance.")
st.markdown("- Explore additional features or embeddings for better representation.")
st.markdown("- Fine-tune the model with a larger dataset for improved generalization.")
st.markdown("- Investigate the use of deep learning models for sentiment analysis.")
