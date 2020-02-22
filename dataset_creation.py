import pandas as pd
import random
import numpy as np
df=pd.read_csv("resturant.csv")
columns=list(df.keys())
print(columns)
columns.remove(columns[0])
columns.remove(columns[0])
print(columns)
d1={}
for i in range(9,10):
    list1=[]
    for j in range(0,1000):
        int1=random.randint(46,49)
        list1.append(int1)
    d1[columns[i]]=list1
weekday=[]
for i in range(0,1000):
    day=random.choice(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    weekday.append(day)
rest=[]
for i in range(0,1000):
    day=random.choice(["A","B","C"])
    rest.append(day)
d1["Restaurant"]=rest
d1["Weekday"]=weekday
print(d1)
finaldata=pd.DataFrame(d1)
print(finaldata.info())
finaldata.to_csv("kichdi.csv")