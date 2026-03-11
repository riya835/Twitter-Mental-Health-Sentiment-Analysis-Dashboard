import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()


def analyze_text(text):
    """Analyze sentiment using VADER."""
    if not isinstance(text, str):
        return 0, 0
    sentiment = sia.polarity_scores(text)
    polarity = sentiment['compound']
    subjectivity = 0.5  # placeholder
    return polarity, subjectivity

def classify_domain(text):
    """Classify mental health domain based on content."""
    if not isinstance(text, str):
        return "General"
    
    keywords = {
        "Depression": ["sad", "hopeless", "depressed", "worthless"],
        "Anxiety": ["worried", "anxious", "nervous", "panic"],
        "Stress": ["stress", "tired", "pressure", "burnout"],
        "Addiction": ["addicted", "can't stop", "social media", "hooked"]
    }
    
    for domain, words in keywords.items():
        if any(word in text.lower() for word in words):
            return domain
    return "General"

def plot_sentiment_trend(df):
    """Plot average sentiment polarity over time."""
    df['date'] = pd.to_datetime(df['date']).dt.date
    sentiment_trend = df.groupby('date')['polarity'].mean()

    fig, ax = plt.subplots()
    ax.plot(sentiment_trend.index, sentiment_trend.values, marker='o', color='purple')
    ax.set_title("Sentiment Trend Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Sentiment Polarity")
    ax.grid()
    return fig

def plot_sentiment_by_domain(df):
    """Bar chart of sentiment across domains."""
    domain_sentiment = df.groupby(['domain', 'sentiment']).size().unstack(fill_value=0)

    fig, ax = plt.subplots()
    domain_sentiment.plot(kind='bar', stacked=True, colormap='coolwarm', ax=ax)
    ax.set_title("Sentiment Distribution Across Domains")
    ax.set_xlabel("Domain")
    ax.set_ylabel("Number of Tweets")
    plt.xticks(rotation=45)
    return fig

def main():
    st.title("Twitter Mental Health Sentiment Analysis")

    # Load both datasets automatically
    try:
        d_tweets = pd.read_csv("clean_d_tweets.csv")
        non_d_tweets = pd.read_csv("clean_non_d_tweets.csv")

        d_tweets["label"] = "Depression"
        non_d_tweets["label"] = "Non-Depression"

        df = pd.concat([d_tweets, non_d_tweets], ignore_index=True)
        st.success("✅ Successfully loaded cleaned datasets!")
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return
    
    # Run sentiment analysis
    df[['polarity', 'subjectivity']] = df['tweet'].apply(lambda x: pd.Series(analyze_text(x)))
    
    # Assign sentiment labels
    df['sentiment'] = df['polarity'].apply(lambda x: "Positive" if x > 0.05 else ("Negative" if x < -0.05 else "Neutral"))
    
    # Classify domains
    df['domain'] = df['tweet'].apply(classify_domain)

    # Display sample data
    st.write("### Sample Analyzed Tweets")
    st.write(df[['tweet', 'polarity', 'sentiment', 'domain', 'label']].head(10))
    
    # Plots
    st.pyplot(plot_sentiment_trend(df))
    st.pyplot(plot_sentiment_by_domain(df))

    # Option to export CSV for Tableau
    st.download_button(
        label="Download Processed Data (CSV)",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="processed_tweets.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
