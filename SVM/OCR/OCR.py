import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
letters=pd.read_csv("letter-recognition.csv")
#print f.head(10)
#print letters["letter"].value_counts().sort_index()
#print letters.iloc[:,1:].describe()
letters_train=letters.iloc[0:14000,]
letters_test=letters.iloc[14000:20000,]
kernels=["rbf","poly","sigmoid"]
for kernel in kernels:
    letter_recognition_model = SVC(C=1, kernel=kernel)
    letter_recognition_model.fit(letters_train.iloc[:, 1:], letters_train["letter"])
    letters_pred = letter_recognition_model.predict(letters_test.iloc[:, 1:])
    # print metrics.classification_report(letters_test["letter"],letters_pred)
    # agreement = letters_test["letter"] == letters_pred
    # print agreement.value_counts()
    print "Accuracy:", metrics.accuracy_score(letters_test["letter"], letters_pred)



