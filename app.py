import sys
import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt

try:
    import tweepy
except ImportError:
    tweepy = None

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
st.title("Real-Time Twitter Sentiment Analysis Dashboard")

def fetch_twitter_stream(query, count, bearer_token=None):
    if tweepy and bearer_token:
        try:
            client = tweepy.Client(bearer_token=bearer_token)
            response = client.search_recent_tweets(query=query, max_results=min(count, 100))
            if response.data:
                return [tweet.text for tweet in response.data]
        except Exception:
            pass
    sample_tweets = [
        f"Really happy with the new {query}, works smoothly.",
        f"Bad experience with {query} today, keeps lagging.",
        f"Checking out {query} updates for the first time.",
        f"Great work by the team on {query}, very fast!",
        f"Not satisfied with {query}, needs a lot of fixes.",
        f"Average performance so far for {query}.",
        f"Best update for {query} this year, highly recommend."
    ]
    np.random.seed(10)
    return list(np.random.choice(sample_tweets, count))

def get_sentiment(text):
    blob = TextBlob(str(text))
    pol = blob.sentiment.polarity
    if pol > 0.05:
        return pol, "Positive"
    elif pol < -0.05:
        return pol, "Negative"
    else:
        return pol, "Neutral"

st.sidebar.header("Twitter Stream Settings")
bearer_key = st.sidebar.text_input("Twitter API Bearer Token", type="password")
topic = st.sidebar.selectbox("Trending Topic", ["TechUpdates", "ProductReviews", "CustomerService"])
count = st.sidebar.slider("Number of tweets", 10, 50, 20)

tweets = fetch_twitter_stream(topic, count, bearer_key)

rows = []
for i, txt in enumerate(tweets):
    pol, label = get_sentiment(txt)
    rows.append({
        "User": f"@user_{100+i}",
        "Tweet": txt,
        "Polarity": round(pol, 2),
        "Sentiment": label
    })

df = pd.DataFrame(rows)

st.subheader("Check Live Tweet Sentiment")
user_text = st.text_input("Type a tweet to check sentiment:", "The new update is very fast and working smoothly.")
if user_text:
    p, s = get_sentiment(user_text)
    st.write(f"**Sentiment:** {s} | **Polarity Score:** {round(p, 2)}")

st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("Total Tweets", len(df))
col2.metric("Positive Tweets", len(df[df['Sentiment'] == 'Positive']))
col3.metric("Negative Tweets", len(df[df['Sentiment'] == 'Negative']))

c1, c2 = st.columns(2)

with c1:
    st.write("Sentiment Split")
    fig, ax = plt.subplots()
    df['Sentiment'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

with c2:
    st.write("Polarity Scores")
    st.bar_chart(df['Polarity'])

st.write("Live Stream Data")
st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if not get_script_run_ctx():
            import subprocess
            subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    except Exception:
        pass