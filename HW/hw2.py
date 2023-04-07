print("Nonlinear Logistic Regression")
print("		or polynomial (degree=2) transform")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

def train_test_split(data,labels,p = 0.5):
	p = int(p*len(labels))
	train_data = data[0:p]
	test_data = data[p-1:-1]
	train_labels = labels[0:p]
	test_labels = labels[p-1:-1]
	return train_data,test_data,train_labels,test_labels

sigmoid = lambda x : 1 / (1 + math.exp(-x))


class LogisticRegression():
	def __init__(self):
		self.w_hat = None
		self.bias = 1
		self.learning_rate = 0.01
	def fit(self,data,labels,max_iterations=500):
		self.w_hat = np.zeros(data.shape[1])
		batch_size = 8
		N = len(labels)
		for epoch in range(max_iterations):
			#print("epoch",epoch)
			for b in range(int(N/batch_size)+1):
				mini_data = data[b*batch_size:min(batch_size*(b+1),N)]
				mini_labels = labels[b*batch_size:min(batch_size*(b+1),N)]
				
				#print(data[b*batch_size:min(batch_size*(b+1),N)])
				#print("		b",b)
				mini_predict = self.predict(mini_data)
				mini_gradient,bias_error = self.error_gradient(mini_data,mini_labels,mini_predict)
				self.w_hat = self.w_hat - self.learning_rate * mini_gradient
				self.bias = self.bias - self.learning_rate * bias_error
				#print(mini_gradient)
			if epoch % 100 == 0:
				print(self.accuracy(labels,self.predict(data)))

	def sigmoid(self,data):
		return 1 / (1 + np.exp(data))
	def binary_cross_entropy(self,true,prediction):
		N = len(true)
		sum = 0
		for i in range(N):
			sum += true[i] * math.log2(prediction[i]) + (1-true) * math.log2(1 - prediction[i])
		return -1/N * sum
	def error_gradient(self,data,true,prediction):
		return np.matmul(  np.transpose(data)  ,( prediction - true )) , sum( prediction - true )

		pass
	def predict(self,data):
		return np.asarray(self.sigmoid([ np.dot(self.w_hat,data[i]) + self.bias for i in range(len(data))]))
		#self.binary_cross_entropy(np.dot(self.w_hat,data) + self.bias)
	def accuracy(self,true,prediction):
		good_count = 0.0
		for i in range(len(true)):
			if true[i] == prediction[i]:
				good_count+=1
			else:
				pass
				#print(true[i],prediction[i])
		return good_count / len(true)

	

# N * d = out dimension
def PolyTransform(data,degree = 2):
	new_data = [] 
	for row in data:
		new_row = []
		for p in range(1,degree+1):
			#print(p)
			for val in row:
				new_val = math.pow(val,p)
				new_row.append(new_val)
				#print(new_val)
		new_data.append(new_row)

	return np.asarray(new_data)


data=pd.read_csv('../lession1/data/heart_disease.csv')
labels=data.values[:,-1]
labels[labels>1]=1
labels=labels.astype(int)

data=data.values[:,:-1]

#Standardize data (substract mean divide with std)
#data= (data - np.mean(data)) / np.std(data)



Tdata = PolyTransform(data,degree = 2)


#print(Tdata)
#quit()

train_data,test_data,train_labels,test_labels = train_test_split(Tdata,labels,0.8)
#train_data,train_labels = (data,labels)
#test_data,test_labels = (data,labels)
lr=LogisticRegression()
lr.fit(train_data,train_labels,10000)
prediction_in=lr.predict(train_data)
prediction_out=lr.predict(test_data)
print("---")
print(lr.accuracy(train_labels,prediction_in))
print(lr.accuracy(test_labels,prediction_out))



#print(Tdata)

quit()





