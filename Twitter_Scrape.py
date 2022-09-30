import re
from re import search
from tkinter.tix import TCL_WINDOW_EVENTS
import tweepy  # Importing of scraper API
import pandas as pd  # Pandas for importing data into DF
from textblob import TextBlob

authen_key = "rAmsI2eOstoA7JnfnAIxKupBI"  # API Consumer key == Your authentication username for scrapong
authen_secret = "xRJoz9kb3WS58KVxM1RwxP0AbRGL7BGaJQz1jwnWuK8klsW58Z"  # Authentication password for scraping

access_token = "1117983115979321345-xv9t5Y0HdlAjpAZrFbGD26d6gY0YEK"  # Access token key
access_token_secret = "tKqckv4J59bO5I7xVrOcCD0buckjOKUg21ZkiHR6CsYOu"  # Access token Secret key

# Pass in the keys for authentication
auth = tweepy.OAuth1UserHandler(
    authen_key, authen_secret,
    access_token, access_token_secret
)

# Start tweepy
api = tweepy.API(auth, timeout=60, wait_on_rate_limit= True)  # Maximum response from twitter = 60 seconds


# This function searches for users and returns the tweets and information
def profile_search(name, counts):
    try:
        # Selects tweets based on username and numbero of tweets
        tweets = tweepy.Cursor(api.user_timeline,screen_name=name).items(counts)
        container = []

        # Gets time, likes, source and text and store them in a list
        for tweet in tweets:
            container.append([tweet.id, tweet.created_at, tweet.favorite_count, tweet.source, tweet.text])  # Nested list

        # Renaming of columns with pandas
        columns = ["id", "Date Created", "Number of Likes", "Source of Tweet", "Tweet"]

        # Creation of Dataframe
        tweets_df = pd.DataFrame(container, columns=columns)
        return tweets_df
    except BaseException as e:
        print('Status Failed On,', str(e))
        exit()


def search_results(search, counts):
    try:
        tweets = tweepy.Cursor(api.search_tweets, q=search, lang= 'en').items(counts) # Search for tweets (only english)
        container = []

        for tweet in tweets:
            container.append([tweet.id ,tweet.created_at, tweet.favorite_count, tweet.source, tweet.text])  # Nested list


            # Renaming of columns with pandas
        columns = ["id","Date Created", "Number of Likes", "Source of Tweet", "Tweet"]

        tweets_df = pd.DataFrame(container, columns=columns)
        tweets_df = tweets_df.drop_duplicates(subset=['Tweet'], keep='last') # Drop duplicates from tweet column
        return(tweets_df)

    except BaseException as e:
        print('Status Failed On,', str(e))
        exit()

# Remove special characters and https
def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) 
    text = re.sub(r'#', '', text) 
    text = re.sub(r'RT\s+', '', text)
    text = re.sub(r'https?:\S+', '', text)
    text = re.sub(r':', '', text)
    text = re.sub(r'_','', text)
    text = re.sub(r'@','', text)
    text = re.sub(r'"', '', text)
    text = " ".join(text.split())
    text = ''.join([c for c in text if ord(c) < 128]) # Only look for ASCII characters


    return text


def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


def popular(text):
    return TextBlob(text).sentiment.polarity


def emotion(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


def export_csv(data):  # Using pandas to export data to csv
    data.to_csv("data.csv")


# Searches for profile / search
# profile_data = profile_search("sporeMOH", 200)

# Search_results for "name" & "number of tweets"
search_data = search_results("healthcare workers", 150)
search_data['Subjectivity'] = search_data['Tweet'].apply(getSubjectivity)  # Adding new column subjectivity
search_data['Polarity'] = search_data['Tweet'].apply(popular)  # Adding new column polarity

search_data['Tweet'] = search_data['Tweet'].apply(clean_text)  # Cleaning of tweets

search_data['Emotion'] = search_data['Polarity'].apply(emotion) # Use polarity to get the emotion

# Prints output
# print(profile_data)
print(len(search_data))
# print(search_data)

# Exports out to csv
export_csv(search_data)
