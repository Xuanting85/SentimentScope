
from tkinter import *
#sets up frame
root = Tk()
root.config(background="light gray")

#variable for each component
words_list = StringVar()
year = StringVar()
diagramtype = IntVar()
#------------------------------------------------------------------------------------
#GUI components

#row 0: title
lblheading = Label(root, text= "Data Search", font=("Ariel",24,"bold")).grid(row=0,column=0,columnspan=4,padx=20,pady=10)

#row 1: key word for twitter hashtags
lblkeyword = Label(root, text="Keyword #", font=("Ariel",14,"bold")).grid(row=1, column=0)
#drop down list of keywords
list1 = ["healthcare", "covid", "hospital", "nurse"]
droplist1 = OptionMenu(root, words_list, *list1)
droplist1.config(width=15)
words_list.set("healthcare")
droplist1.grid(row=1, column=2)

#row 2: year
lblyear = Label(root, text="year :", font=("Ariel",14,"bold")).grid(row=2, column=0)
list2 = ["2020", "2021", "2022", "2020-2022"]
droplist2 = OptionMenu(root, year, *list2)
droplist2.config(width=15)
year.set("2020")
droplist2.grid(row=2, column=2)

#row 3: display types
lbldisplay = Label(root, text="Data type :", font=("Ariel",14,"bold")).grid(row=3, column=0)
selectbutton1 = Radiobutton(root, text="piechart", variable=diagramtype, value=1, font=("Ariel",14,"bold")).grid(row=3,column=1)
selectbutton2 = Radiobutton(root, text="bar", variable=diagramtype, value=2, font=("Ariel",14,"bold")).grid(row=3,column=2)

display_button = Button(root, text="Display").grid(row=7,column=1,sticky=E, padx=20, pady=10)

root.mainloop()