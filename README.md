# Sentiment-Analysis-Of-Lockdown-In-India-During-Covid-19



## Introduction

This project analyzes sentiments from Twitter data during the lockdown in India during the Covid-19 pandemic. The project leverages Natural Language Processing (NLP) techniques to classify sentiments (e.g., happy, sad, anger) based on textual data. The analysis is implemented using a Naive Bayes classifier and is hosted in a Streamlit application.

The goal of this project is to understand public sentiment during a significant period of social restriction and to explore potential improvements and future applications of sentiment analysis in social contexts.

## Project Structure

- **main.py**: The main script containing the Streamlit application code.
- **finalSentimentdata2.csv**: The dataset containing tweets and their corresponding sentiments.
- **sentiment_model.pkl**: The trained Naive Bayes model, along with the label encoder and TF-IDF vectorizer.
- **requirements.txt**: A file listing the Python dependencies required to run the project.

## Features

- **Sentiment Prediction**: Enter a keyword to predict the sentiment of related tweets.
- **Evaluation Results**: View the accuracy, classification report, and confusion matrix for the model.
- **Visualizations**: Display an image related to the lockdown and a heatmap of the confusion matrix.
- **Recommendations and Future Work**: Suggestions for improving the model and extending the project.

## How to Use

### 1. Clone the Repository

First, clone the repository to your local machine:

```sh
git clone https://github.com/yourusername/Sentiment-Analysis-Of-Lockdown-In-India-During-Covid-19.git
cd Sentiment-Analysis-Of-Lockdown-In-India-During-Covid-19
```
# Sentiment Analysis of Lockdown in India During Covid-19

## How to Use

### 2. Install the Requirements

Install the necessary Python packages using pip:

```sh
pip install -r requirements.txt
```
### 3. Run the Application

Run the Streamlit app:

```sh
streamlit run main.py
```

