# -*- coding: utf-8 -*-
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
iris=pd.read_csv("iris.csv")
#print iris.head(10)

#sns.pairplot(iris,hue="Class")
#plt.show()#发现红绿两类难以区分

class_dict={"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}
iris["Class"].map(class_dict)
#print iris["Class"].value_counts()
y=iris["Class"]
X=iris.drop("Class",axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
#print y_train.value_counts()
#print y_test.value_counts()
classfier=LogisticRegression(C=1000.0,solver='lbfgs')
classfier.fit(X_train,y_train)
y_pre=classfier.predict(X_test)
#print metrics.classification_report(y_test,y_pre)
#print metrics.accuracy_score(y_test,y_pre)
Metrics=metrics.confusion_matrix(y_test,y_pre)
#sns.heatmap(Metrics,annot=True,fmt="d")
#plt.show()
coe_df=pd.DataFrame(classfier.coef_,columns=iris.columns[0:4])
#print coe_df.round(2)
coe_df["jieju"]=classfier.intercept_
print coe_df
print "这个矩阵对应的系数就是逻辑回归的系数"
#http://hackdata.cn/note/view_static_note/2dc6f02da4592f96a4c888d1408ed604/