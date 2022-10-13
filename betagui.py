import PySimpleGUI as sg
import Sentimental
import pandas as pd
df = Sentimental.data_read_clean(pd.read_csv('data.csv'))
sg.theme('BluePurple')
col_1 = [[sg.OK("Display"),sg.Cancel()]]
col_2 = [[sg.Radio("Piechart", "group 1",size=[30,1], key="-CB1-")],[sg.Radio("Histogram", "group 1",size=[20,1] , key="-CB2-")],
[sg.Radio("Kernal graph","group 1",size=[20,1], key="-CB3-")], [sg.Radio("Scatter Graph","group 1",size=[20,1], key="-CB4-")]]
layout = [[sg.Col(col_2)], [sg.Col(col_1)]]

window = sg.Window("Data search", layout, size=(250, 200))

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == "Cancel":
        break
    elif event == "Display":
        if values['-CB1-'] == True:
            Sentimental.pie_chart(df)
        elif values['-CB2-'] == True:   
            Sentimental.histo(df)
        elif values['-CB3-'] == True:
            Sentimental.kernal_graph(df)
        elif values['-CB4-'] == True:
            Sentimental.scatter_plot(df)
#print(event,values)
