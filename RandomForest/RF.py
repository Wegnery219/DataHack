import numpy as np
import pandas as pd

train_x=pd.read_csv("train_x.csv")
train_y=pd.read_csv("train_y.csv")
#print train_x.head(2)
value_missing=train_x.values.ravel()
print (len(value_missing[value_missing==np.nan]))
test_array=np.array([[1,2,3],[4,5,6]])
t_diff=np.diff(test_array,axis=0)
print t_diff