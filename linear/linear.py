# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
boston=load_boston()
print boston.keys()
bos=pd.DataFrame(boston.data,columns=boston.feature_names)
#print bos.head()
bos["PRICE"]=boston.target
lm=LinearRegression()
X=bos.drop("PRICE",axis=1)
lm.fit(X,bos["PRICE"])
# print "截距为：",lm.intercept_
# print "回归系数为：",lm.coef_
# print pd.DataFrame(zip(X.columns,lm.coef_),columns=["特征","回归系数"])
# plt.scatter(bos.RM,bos.PRICE)
# plt.xlabel("RM")
# plt.ylabel("PRICE")
# plt.show()
#

# lm.predict(X)[0:5]
plt.scatter(bos.PRICE,lm.predict(X))
plt.xlabel("Prices")
plt.ylabel("P_prices")
plt.show()