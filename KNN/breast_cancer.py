import pandas as pd
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
def min_max_normolize(x):
    return (x-x.min())/(x.max()-x.min())
breast_cancer=pd.read_csv("wdbc.csv")
del breast_cancer["id"]
#print breast_cancer.head(5)
#print breast_cancer.diagnosis.value_counts()
diagnosis_dict={"B":0,"M":1}
breast_cancer["diagnosis"]=breast_cancer["diagnosis"].map(diagnosis_dict)
#print breast_cancer.diagnosis.value_counts()
for col in breast_cancer.columns[1:31]:
    breast_cancer[col]=min_max_normolize(breast_cancer[col])
#print breast_cancer.iloc[:,1:].describe()
y=breast_cancer["diagnosis"]
del breast_cancer["diagnosis"]
X=breast_cancer
bc_train,bc_test,bc_trainlable,bc_testlable=cross_validation.train_test_split(X,y,test_size=0.3,random_state=0)
#print bc_trainlable.value_counts()/len(bc_trainlable)
#print bc_testlable.value_counts()/len(bc_testlable)
knn_model=KNeighborsClassifier(n_neighbors=21)
knn_model.fit(bc_train,bc_trainlable)
knn_modelpre=knn_model.predict(bc_test)
print metrics.classification_report(bc_testlable,knn_modelpre)
print metrics.confusion_matrix(bc_testlable,knn_modelpre)
print metrics.accuracy_score(bc_testlable,knn_modelpre)
