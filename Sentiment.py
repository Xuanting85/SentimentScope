from optparse import Values
import tweepy #access twt API
from textblob import TextBlob 
import pandas as pd
import numpy as np
import re #regular expression library 
import matplotlib.pyplot as plt



#get twt using tweepy & cursor, exlude replies & amt of twts we want
tweets = tweepy.Cursor (twitterApi.user_timeline,
                        screen_name=twitterAccount,
                        count=None,
                        since_id=None,
                        max_id=None, trim_user=True, exclude_replies=True, contribubtor_details=False,
                        include_entities=False).items(); #how many tweets we want to refer
df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweet'])
df.head()

#i just want the tweet itself that can assist us in the analysis
#function that helps to remove mentions and all not needed stuff using re
def cleanUpTweet(txt):
    txt = re.sub(r'@[A-Za-z0-9_]+', '', txt) #remove mentions
    txt = re.sub(r"#", '', txt) #remove hashtags
    txt = re.sub(r'RT : ', '', txt) #remove RTS
    txt = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+', '', txt) #remove URLS
    return txt

df['Tweet']=df['Tweet'].apply(cleanUpTweet) #apply function to all twts


def getTextSubjectivity(txt):
    return TextBlob(txt).sentiment.subjectivity

def getTextPolarity(txt):
    return TextBlob(txt).sentiment.polarity

df['Subjectivity']=df['Tweet'].apply(getTextSubjectivity)
df['Polarity']=df['Tweet'].apply(getTextPolarity)

df = df.drop(df[df['Tweet']==''].index)
df.head() #amount of tweets

#get text analysis
def getTextAnalysis(a): #based on polarity score
    if a<0:
        return "Negative"
    elif a==0:
        return "Neutral"
    else:
        return "Positive" 

df["Score"]=df['Polarity'].apply(getTextAnalysis) #store dataframe based by score
df.head() #to see the df

#extract % on sentiment analysis
positive =df[df['Score'] == "Positive"]
print(str(positive.tweet[0]/(df.tweet[0])*100)+"% of positive tweets")
pos = positive.tweet[0]/df.tweet[0]*100

negative =df[df['Score'] == "Negative"]
print(str(negative.tweet[0]/(df.tweet[0])*100)+"% of negative tweets")
neg = negative.tweet[0]/df.tweet[0]*100

neutral =df[df['Score'] == "Neutral"]
print(str(neutral.tweet[0]/(df.tweet[0])*100)+"% of neutral tweets")
neutrall = neutral.tweet[0]/df.tweet[0]*100

#PieChart
explode=(0,0.1,0)
labels= 'Positive', 'Negative', 'Neutral'
np.size[pos,neg,neutrall] #help
colors=['green','red','blue']
plt.pie(np.size,explode=explode, colors=colors, autopct='%1.1f%%', startangle=120)
plt.legend(labels, loc = (-0.05, 0.05), shadow=True)
plt.axis('equal')
plt.savefig("Sentimental Analysis.png")

#Plot Polarity Vs Subjectivity BarChart
labels = df.groupby('Score').count().index.values
values = df.groupby('Score').size().values
plt.bar(labels,values) 

#plot polarity & subjectivity
#or index, row in df.iterrows():
 #   if row['Score']=='Positive':
  #      plt.scatter(row[Polarity],row[Subjectivi])