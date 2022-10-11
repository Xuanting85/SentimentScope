import tkinter as tk
from Sentimental import data_read_clean
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

root = tk.Tk() # Defining the start of the GUI
root.geometry("1200x800") # Height of the GUI
root.title("Python Covid Analysis") # Title of the GUI

df = data_read_clean(pd.read_csv('data.csv'))  # Read data from csv and drop duplicates from column "Tweet"
title_label = tk.Label(root, text="Data Analysis with Python", font=('Times New Roman', 20))
title_label.pack(pady=40)


def pie_chart(df):     # Creating a pie chart with % to show counts with number of likes
    lables = df['Emotion']
    values = df['Number of Likes']
    fig = go.Figure(data=[go.Pie(labels=lables, values=values)])
    fig.update_layout(
    title_text="Percentage of emotion and likes") 
    fig.show()


pie_button = tk.Button(root, text="Pie Chart", command = pie_chart(df))
pie_button.pack()

root.mainloop()