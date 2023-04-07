
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data=pd.read_csv('D:/Programing/Gitted/MachineLearning/lession2/data/heart_disease.csv')
labels=data.values[:,-1]
labels[labels>1]=1
labels=labels.astype(int)

data=data.values[:,:-1]

#Standardize data (substract mean divide with std)
#data=

def visualize(data,labels):
    pass
    #TODO



class LogisticRegression():
    def __init__(self):
        self.w_hat = None
    def fit(self,data,labels,max_iterations=500):
        #TODO
        #self.w_hat =
        pass

    def sigmoid(self,data):
        #TODO
        pass
    def binary_cross_entropy(self,true,prediction):
        #TODO
        pass
    def error_gradient(self,data,true,prediction):
        #TODO
        pass
    def predict(self,data,w_hat):
        #TODO
        pass
    def accuracy(self,true,prediction):
        #TODO
        pass
 #train_data,test_data,train_labels,test_labels = train_test_split(data,labels)
lr=LogisticRegression()
lr.fit(train_data,train_labels)
prediction_in=lr.predict(train_data)
prediction_out=lr.predict(test_data)
print(lr.accuracy(train_labels,prediction_in))
print(lr.accuracy(test_labels,prediction_out))


