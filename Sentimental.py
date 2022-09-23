import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import plotly.express as px

df = pd.read_csv('data.csv') # Read data from csv


# Creates a histogram based on the number of likes for each tweet
fig = px.histogram(df, x="Number of Likes") #Takes data from the column "Likes"
fig.update_traces(marker_color="red",marker_line_color='rgb(1,12,100)',
                  marker_line_width=1)
fig.update_layout(title_text='Likes from tweets')
fig.show()


# print(df[df['Number of Likes'] > 5]) # Printing tweets with more than 5 likes

df.isna().sum
df.dropna(inplace=True) # Drop any null values
print(df.head) # Print out the headers 
df['Tweet'] = df['Tweet'].str.lower() # Convert tweets to lower caps

# Creating a wordcloud
stopwords = set(STOPWORDS)
stopwords.update(["br", "href"])
words = " ".join(tweets for tweets in df.Tweet)
wordcloud = WordCloud(width = 500, height = 500, 
background_color = "white", stopwords=stopwords, min_font_size=10).generate(words)
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('wordcloud10.png')
plt.show()