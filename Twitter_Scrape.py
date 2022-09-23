from re import search
import tweepy # Importing of scraper API
import pandas as pd # Pandas for importing data into DF


authen_key = "rAmsI2eOstoA7JnfnAIxKupBI" #API Consumer key == Your authentication username for scrapong
authen_secret = "xRJoz9kb3WS58KVxM1RwxP0AbRGL7BGaJQz1jwnWuK8klsW58Z" # Authentication password for scraping

access_token = "1117983115979321345-xv9t5Y0HdlAjpAZrFbGD26d6gY0YEK"    #Access token key
access_token_secret = "tKqckv4J59bO5I7xVrOcCD0buckjOKUg21ZkiHR6CsYOu" #Access token Secret key

#Pass in the keys for authentication
auth = tweepy.OAuth1UserHandler(
    authen_key, authen_secret,
    access_token, access_token_secret
)

#Start tweepy
api = tweepy.API(auth, timeout= 60) # Maximum response from twitter = 60 seconds

# This function searches for users and returns the tweets and information
def profile_search(name, count_tweets):
    
    try:
        # Selects tweets based on username and numbero of tweets
        tweets = api.user_timeline(screen_name = name,count = count_tweets)
        container = []
        
        # Gets time, likes, source and text and store them in a list
        for tweet in tweets:
            container.append([tweet.created_at, tweet.favorite_count, tweet.source,  tweet.text]) #Nested list 
     
        # Renaming of columns with pandas
        columns = ["Date Created", "Number of Likes", "Source of Tweet", "Tweet"]
        
        #Creation of Dataframe
        tweets_df = pd.DataFrame(container, columns=columns)
        return tweets_df
    except BaseException as e:
        print('Status Failed On,',str(e))
        exit()


def search_results(search, count_tweets):
   
    try:
        tweets = api.search_tweets(q = search, count = count_tweets)
        container = []

        for tweet in tweets:
            container.append([tweet.created_at, tweet.favorite_count, tweet.source,  tweet.text]) # Nested list

        
        # Renaming of columns with pandas
        columns = ["Date Created", "Number of Likes", "Source of Tweet", "Tweet"]

        tweets_df = pd.DataFrame(container, columns=columns)
        return tweets_df
    except BaseException as e:
        print('Status Failed On,',str(e))
        exit()


def export_csv(data): # Using pandas to export data to csv
    data.to_csv("data.csv")

# Searches for profile / search
profile_data = profile_search("MOH", 100)
search_data = search_results("healthcare", 100)

#Prints output
#print(profile_data)
print(search_data)

#Exports out to csv
export_csv(search_data)
