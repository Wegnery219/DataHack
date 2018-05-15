import pandas as pd
credit=pd.read_csv("german.csv")
#print credit.head(5)
print credit.default.value_counts()