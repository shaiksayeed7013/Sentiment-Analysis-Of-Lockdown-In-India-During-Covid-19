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
### 4. Predict Sentiments

Enter a keyword related to Covid-19 or the lockdown to see the predicted sentiment.

View evaluation metrics like accuracy and classification report.

Analyze the confusion matrix to understand the model's performance.

### 5. Explore and Modify

You can customize the model, the data preprocessing, or the Streamlit app to suit your needs:


Data Preprocessing: Modify the LabelEncoder, TfidfVectorizer, or the dataset.

Model Training: Experiment with different models or hyperparameters.

App Features: Add new features or improve the user interface in main.py.

## Future Work

Advanced Models: Explore more sophisticated models for sentiment analysis, such as BERT or LSTM.

Hyperparameter Tuning: Experiment with different hyperparameters to enhance model accuracy.

Data Augmentation: Use additional data or techniques like embeddings to improve feature representation.

Deep Learning: Implement deep learning models to capture complex sentiment patterns.

## Overview

The "Sentiment Analysis of Lockdown in India During Covid-19" project aims to analyze public sentiment on Twitter during the lockdown period in India due to the Covid-19 pandemic. The project utilizes Natural Language Processing (NLP) techniques to classify sentiments (e.g., happy, sad, anger) based on tweet text. The analysis is implemented using a Naive Bayes classifier and is hosted in a Streamlit application.

## Live Demo

Check out the live demo of the project here: [Sentiment Analysis of Lockdown in India during COVID-19](https://sentiment-analysis-of-lockdown-in-india-during-covid-19-2kbdxw.streamlit.app/)

## Useful Links

- [Streamlit Documentation](https://docs.streamlit.io)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
