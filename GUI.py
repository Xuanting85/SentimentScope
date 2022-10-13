import PySimpleGUI as sg
import snscrape.modules.twitter as sntwitter # Importing of Scraper API
import re
import pandas as pd  # Pandas for importing data into DF
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Vader analysis returns the polarity of the comment
from collections import Counter
from distutils.command.config import dump_file
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import plotly.express as px
pd.options.mode.chained_assignment = None  # default='warn
from textblob import TextBlob
from nltk.corpus import stopwords
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


def clean_text(texts): # Remove special characters and https
    texts = re.sub(r'@[A-Za-z0-9]+', '', texts) 
    texts = re.sub(r'#', '', texts) 
    texts = re.sub(r'RT\s+', '', texts)
    texts = re.sub(r'https?:\S+', '', texts)
    texts = re.sub(r':', '', texts)
    texts = re.sub(r'_','', texts)
    texts = re.sub(r'@','', texts)
    texts = re.sub(r'"', '', texts)
    texts = " ".join(texts.split())
    texts = ''.join([c for c in texts if ord(c) < 128]) # Only look for ASCII characters
    return texts


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

# The analysis available are piechart / histogram / wordcloud / kernal graph / time graph / scatterplot 

def getSubjectivity(text): # Returns the subjectivity from the library textblob
    return TextBlob(text).sentiment.subjectivity


def getPolarity(text): # Returns the polarity from the library textblob
    return TextBlob(text).sentiment.polarity


def data_read_clean(df): # Perform further data cleaning
    cols = [0] # Specficy first column to drop
    df = df.drop(df.columns[cols], axis=1) # Drop first 2 columns which are unncessary
    df['Tweet'] = df['Tweet'].str.lower()  # Convert tweets to lower caps
    df = df.drop_duplicates(subset=['Tweet'], keep='last') # Drop duplicates from tweet column

    remove_words = ["healthcare", "Healthcare", "Worker", "worker", "healthcare worker", "workers", "never", "so", "before", 
    "healthcare workers", "the", "to", "and", "of", "for", "a", "in", "is", "are", "that", "on", "you", "with", "amp"] # Specify common words to be removed4

    rem = r'\b(?:{})\b'.format('|'.join(remove_words)) # Set parameters to remove this list of words from "Tweet" column
    df['Tweet'] = df['Tweet'].str.replace(rem, '') # Apply the removal to the pandas dataframe tweet

    df['Subjectivity'] = df['Tweet'].apply(getSubjectivity)  # Adding new column subjectivity from textblob
    df['Polarity'] = df['Tweet'].apply(getPolarity)  # Adding new column Polarity from textblob
    return df


# Functions below perform visualization with different charts / graphs

def pie_chart(df):     # Creates a pie chart to count % of each emotion
    emotions = df['Emotion'].value_counts()
    wp={'linewidth':2, 'edgecolor': 'black'}
    explode = (0.1,0.1,0.1)
    emotions.plot.pie(y=emotions, subplots=True, figsize=(5,5),colors=['green','red','blue'], autopct='%1.0f%%',shadow = True, wedgeprops=wp
    ,explode = explode, label='')
    plt.title("Polarity Distribution")
    plt.show()

    
def histo(df): # Creates a histogram based on the number of likes for each tweet
    fig = px.histogram(df, x="Emotion", y ="Number of Likes", title="Number of likes for each emotion", width=1200, height=1000) # Takes data from the column "Number of Likes"
    fig.update_traces(textfont_size = 100,
                    marker_line_width=1, marker_color=["blue", "red", "green"])
    fig.show()


def wordcloud(tweet, title, col): # Creating a wordcloud with different emotions
    # image = np.array(Image.open('hashtag.png'))
    stopwords = set(STOPWORDS)
    stopwords.update(["br", "href"])
    words = " ".join(tweets for tweets in tweet.Tweet)
    wordcloud = WordCloud(width=1000, height=800, 
                        background_color="white", stopwords=stopwords, min_font_size=10, colormap=col).generate(words)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, size = 20)
    plt.show()


def kernal_graph(df):# Kernal distribution graph
# Estimate density of the distribution of emotion / similar to a histogram
    num_neg = df[df['Emotion']=='Negative']['Tweet'].apply(lambda x: len(x.split()))
    num_neu = df[df['Emotion']=='Neutral']['Tweet'].apply(lambda x: len(x.split()))
    num_pos = df[df['Emotion']=='Positive']['Tweet'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(12,6))
    sns.kdeplot(num_neg, fill=True, color = 'r').set_title('Distribution of number of words')
    sns.kdeplot(num_neu, fill=True, color = 'y')
    sns.kdeplot(num_pos, fill=True, color = 'b')

    plt.legend(labels=['Negative', 'Neutral','Positive'])
    plt.show()


def time_bar(start, end, df): # Graph to show sentiments over time
    df['Date Created'] = pd.to_datetime(df['Date Created'])
    df['Date Created'] = df['Date Created'].dt.date
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    mask = (df['Date Created'] > start) & (df['Date Created'] <= end)
    df2 = df.loc[mask]
    grouping = df2.groupby(by='Date Created')['Emotion'].value_counts()
    unstack_graph = grouping.unstack(level=1)
    unstack_graph.plot.bar()
    plt.show()


def scatter_plot(df): # Scatter plot between subjectivity & polarity
    df.plot.scatter(x="Polarity", y="Subjectivity", c="DarkBlue", colormap="viridis")
    plt.show()
    

def most_common(df): # Barplot to show the count of popular words
    sentences = [] # Create a new list to store sentences from tweets
    for word in df['Tweet']:
        sentences.append(word)
    sentences
    # print(sentences[:10])

    lines = []
    for line in sentences:
        words = line.split() # Split this sentences up to get each word
        for w in words:
            lines.append(w) # Append the words to a new list
    # print(lines[:10])

    stop_words = set(stopwords.words('english'))
    # print(stop_words)
    new = []
    for w in lines:
        if w not in stop_words: # Remove unwanted words using stoplist
            new.append(w)

    # print(new[:10])

    df_count = pd.DataFrame(new)
    # Further removal of punctuations
    df_count.drop(df_count[df_count[0] == '.'].index, inplace=True)
    df_count.drop(df_count[df_count[0] == ','].index, inplace=True)
    df_count.drop(df_count[df_count[0] == '&;'].index, inplace=True)
    df_count.drop(df_count[df_count[0] == '-'].index, inplace=True)
    df_count = df_count[0].value_counts() # Count the occurence of each word


    df_count = df_count[:20] # Take the first 20 words with the most number of count
    plt.figure(figsize=(10,5))
    sns.barplot(df_count.values, df_count.index, alpha=0.8)
    plt.title('Top Words Overall')
    plt.xlabel('Count of words', fontsize=12)
    plt.ylabel('Word from Tweet', fontsize=12)
    plt.show() 


def open_data_window(df):
    df['Tweet'] = df['Tweet'].apply(clean_text)  # Cleaning of tweets
    print(df)
    df = data_read_clean(df)  # Read data from csv and drop duplicates from column "Tweet"
    print(df)
    layout = [[sg.Text("Data Analysis Window\n\nPlease select one of the analysis below", key="new")],
    [sg.Button("Piechart"),sg.Button("Histogram"), sg.Button("Kernal Graph")],
    [sg.Button("Positive Word Cloud"), sg.Button("Negative Word Cloud"), sg.Button("Neutral Word Cloud")],
    [sg.Button("Scatter"), sg.Button("Time Graph"), sg.Button("Most Common Words")]]
    window = sg.Window("Second Window", layout, modal=True) # Unable to interact with main window until you close second window
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        elif event == "Piechart":
            pie_chart(df)

        elif event == "Histogram":
            histo(df)
        elif event == "Kernal Graph":
            kernal_graph(df)

        elif event == "Positive Word Cloud":
            df_positive = df.loc[df['Emotion'] == "Positive"] # Selecting columns with positive emotion
            wordcloud(df_positive, "Positive Word Cloud", "Greens")
        elif event == "Negative Word Cloud":
            df_negative = df.loc[df['Emotion'] == "Negative"] # Selecting columns with negative emotion
            wordcloud(df_negative, "Negative Word Cloud", "Reds")
        elif event == "Neutral Word Cloud":
            def_neutral = df.loc[df['Emotion'] == "Neutral"] # Selecting columns with neutral emotion
            wordcloud(def_neutral, "Neutral Word Cloud", "Blues")

        elif event == "Scatter":
            print(df)
            scatter_plot(df)

        elif event == "Time Graph":
            time_bar('2020-01-01','2021-01-01', df)

        elif event == "Most Common Words":
            most_common(df)

    window.close()

date_time = "since:2020-02-01 until:2020-05-01" # Scrape from Feb to May 2020
list1 = ["healthcare workers ", "covid ", "nurse ", "hospital ", "doctor "]
layout = [[sg.Text('SCRAPER \nPlease select the keyword to scrape from twitter')],
 [sg.Text('Scraper', size =(15, 2)), sg.DD(list1, key = "key_word")],
 [sg.Text('Amount to Scrape', size =(15, 2)), sg.InputText(key = "number")],
    [sg.Exit(), sg.Button("Scrape Data"), sg.Button("Export to CSV")],
    [sg.Button("Data Analysis")]]


window = sg.Window("Python Analysis", layout)

while True:
    try:
        event, values = window.read()
        # search_data = pd.DataFrame()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break
        elif event == "Scrape Data": # Scrapes data and stores it in search_data
            search_data = search_results(int(values["number"]), values["key_word"]+ date_time)
            search_data['Polarity'] = search_data['Tweet'].apply(popular)  # Adding new column polarity using VaderSentiment Analysis
            search_data['Emotion'] = search_data['Polarity'].apply(emotion) # Use polarity to get the emotion
            if search_data.empty:
                sg.popup_auto_close("Scrape unsuccessful")
            elif not search_data.empty:
                sg.popup_auto_close("Scrape successful")

        search_data = search_data

        if event == "Export to CSV": # Allows one to export search_data to a csv file
            if search_data.empty:
                sg.popup_auto_close("Export unsuccessful")
            elif not search_data.empty:
                sg.popup_auto_close("Export successful")
                export_csv(search_data)

        if event == "Data Analysis": # Data must be scraped first before Data Analysis can be opened
            open_data_window(search_data)
    except:
        sg.popup_auto_close("Error encountered, please try again") 

window.close()
