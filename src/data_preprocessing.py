import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    sentiment = sid.polarity_scores(text)
    return sentiment['compound']

# the csv file contains daily stock market news headlines (25 headlines per date), 
# a corresponding date, and a binary label indicating the market direction 
# (1 for non-dropping [rising or stable], 0 for dropping, which is changed to -1 to be consistent for the adaboost algorithm)
# non-dropping is "bullish", dropping is "bearish"

def load_dataset():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    csv_path = os.path.join(data_dir, "Combined_News_DJIA.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    df = pd.read_csv(csv_path)
    return df

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    cleaned_text = " ".join(filtered_tokens)
    return cleaned_text

def combine_headlines(row):
    headlines = row[2:]

    combined = " ".join([str(headline) for headline in headlines if pd.notnull(headline)])
    return combined

def process_data():
    df = load_dataset()

    df["Combined_News"] = df.apply(combine_headlines, axis=1)

    # download necessary NLTK packages and clean the text
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    df["Cleaned_News"] = df["Combined_News"].apply(clean_text)

    # calculate sentiment score for each day
    df["Sentiment_Score"] = df["Cleaned_News"].apply(get_sentiment_score)

    # convert labels to binary format
    df["Label_Binary"] = df["Label"].apply(lambda x: 1 if x == 1 else -1)

    # convert date column to datetime type and split data into train and test sets
    df["Date"] = pd.to_datetime(df["Date"])
    train_df = df[(df["Date"] >= "2008-08-08") & (df["Date"] <= "2014-12-31")]
    test_df = df[(df["Date"] >= "2015-01-02") & (df["Date"] <= "2016-07-01")]

    # vectorize the cleaned text using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1500, min_df=5, stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(train_df["Cleaned_News"]).toarray()
    X_test_tfidf = vectorizer.transform(test_df["Cleaned_News"]).toarray()

    # extract sentiment scores and reshape
    train_sentiment = train_df["Sentiment_Score"].values.reshape(-1, 1)
    test_sentiment = test_df["Sentiment_Score"].values.reshape(-1, 1)

    # standardize the sentiment scores
    scaler = StandardScaler()
    train_sentiment_scaled = scaler.fit_transform(train_sentiment)
    test_sentiment_scaled = scaler.transform(test_sentiment)

    # append sentiment features to the TF-IDF features
    X_train = np.hstack((X_train_tfidf, train_sentiment_scaled))
    X_test = np.hstack((X_test_tfidf, test_sentiment_scaled))

    y_train = train_df["Label_Binary"].values
    y_test = test_df["Label_Binary"].values

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = process_data()
    print("Training features shape:", X_train.shape)
    print("Testing features shape:", X_test.shape)
    print("Sample training labels:", y_train[:5])
    