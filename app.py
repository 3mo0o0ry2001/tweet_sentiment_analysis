import streamlit as st
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re

# تحميل النموذج
with open("sentiment_model.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

# تنظيف النصوص
def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub(r'http[s]?://\S+', '', token)
        token = re.sub(r'(@[A-Za-z0-9_]+)', '', token)
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

stop_words = stopwords.words('english')

# واجهة التطبيق
st.title("Tweet Sentiment Analysis")
tweet = st.text_area("Enter a tweet:")

if st.button("Analyze Sentiment"):
    if tweet:
        tokens = word_tokenize(tweet)
        cleaned_tokens = remove_noise(tokens, stop_words)
        sentiment = classifier.classify(dict([token, True] for token in cleaned_tokens))
        st.write(f"The sentiment of the tweet is: **{sentiment}**")
    else:
        st.write("Please enter a tweet to analyze.")

