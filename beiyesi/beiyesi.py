# -*- coding:utf-8 -*-
import pandas as pd
from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
# s = u'我想和女朋友一起去北京故宫博物院参观和闲逛。'
#
# cut = jieba.cut(s)
#
# print '【Output】'
# print cut
# print ','.join(cut)
train_x=pd.read_csv("train_x.csv")
train_y=pd.read_csv("train_y.csv")
test_x=pd.read_csv("test_x.csv")
x=train_x[list(train_x.columns.values)[0]].map(lambda xx:' '.join((jieba.cut(xx))))
list_text=list(x)
vectorizer=TfidfVectorizer()
wordar=vectorizer.fit_transform(list_text)

# print train_x.head(1)
# del train_x[0]
# vectorizer=TfidfVectorizer()
# wordar=vectorizer.fit_transform(train_x)
# words=vectorizer.get_feature_names()
# print words[0]
# print wordar[0]
clf=naive_bayes.MultinomialNB(alpha=1.0,fit_prior=True,class_prior=None)
clf.fit(wordar,train_y)


