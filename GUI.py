import PySimpleGUI as sg
from attr import attributes
import snscrape.modules.twitter as sntwitter # Importing of Scraper API
import re
import pandas as pd  # Pandas for importing data into DF
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Vader analysis returns the polarity of the comment
# Eg how neg or pos a comment this and the compound is the overall 
# nltk.downloader.download('vader_lexicon') # Remember to uncomment this to install lexicon file before running scraper

def search_profile(counts, profiles): # Using Snscrape API to scrape profile data on twitter
    attributes_container = []
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(profiles).get_items()):
        if i>counts:
            break
        attributes_container.append([tweet.date, tweet.likeCount, tweet.sourceLabel, tweet.content])
    tweets_df = pd.DataFrame(attributes_container, columns=["Date Created", "Number of Likes", "Source of Tweet", "Tweet"])
    return tweets_df


def search_results(counts, topic_time): # Using Snscrape API to scrape search result data
    attributes_container = []
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(topic_time).get_items()): # For each tweet in given topic / time
        if i>counts:
            break
        attributes_container.append([tweet.user.username, tweet.date, tweet.likeCount, tweet.sourceLabel, tweet.content])
    # Creating a dataframe to load the list
    tweets_df = pd.DataFrame(attributes_container, columns=["User", "Date Created", "Number of Likes", "Source of Tweet", "Tweet"])
    return tweets_df


def clean_text(text): # Remove special characters and https
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


def popular(text):
    vader = SentimentIntensityAnalyzer()
    return vader.polarity_scores(text)


def emotion(score):
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def export_csv(data):  # Using pandas to export data to csv
    data.to_csv("data_sample.csv")


def append_csv(data): # Append data to csv
    data.to_csv("data_test.csv", mode='w', header=False)


def clear_csv(file):
    f = open(file, 'w')
    f.truncate()
    f.close()

layout = [[sg.Text('FORMAT \n(keyword since:yyyy-mm-dd until: yyyy-mm-dd)')],
 [sg.Text('Scraper', size =(15, 2)), sg.InputText(key = "key_word")],
 [sg.Text('Amount to Scrape', size =(15, 2)), sg.InputText(key = "number")],
    [sg.Exit(), sg.Button("Scrape Data"), sg.Button("Export to CSV")]]


window = sg.Window("Python Analysis", layout)

while True:
    try:
        event, values = window.read()
        # search_data = pd.DataFrame()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break
        elif event == "Scrape Data":
            search_data = search_results(int(values["number"]), values["key_word"])
            search_data['Polarity'] = search_data['Tweet'].apply(popular)  # Adding new column polarity using VaderSentiment Analysis
            search_data['Tweet'] = search_data['Tweet'].apply(clean_text)  # Cleaning of tweets
            search_data['Emotion'] = search_data['Polarity'].apply(emotion) # Use polarity to get the emotion
            if search_data.empty:
                sg.popup_auto_close("Scrape unsuccessful")
            elif not search_data.empty:
                sg.popup_auto_close("Scrape successful")
        search_data = search_data
        print(search_data)
        if event == "Export to CSV":
            if search_data.empty:
                sg.popup_auto_close("Export unsuccessful")
            elif not search_data.empty:
                sg.popup_auto_close("Export successful")
                export_csv(search_data)
    except:
        sg.popup_auto_close("Error encountered, please try again")

window.close()
