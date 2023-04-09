
import math

# NOTE This is the Hoeffding example (task1)

print("Hoeffding example:")

N = 100.0
E_in = 0.1
E_out = 0.2
e = abs(E_in-E_out)

bound = 2*math.exp(-2*math.pow(e,2)*N)

print("		the bound is {}".format(bound))

print("Nonlinear Logistic Regression")
print("		or polynomial (degree=2,3) transform")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression

def train_test_split(data,labels,p = 0.5):
	p = int(p*len(labels))
	train_data = data[0:p]
	test_data = data[p-1:-1]
	train_labels = labels[0:p]
	test_labels = labels[p-1:-1]
	return train_data,test_data,train_labels,test_labels

def accuracy(true,prediction):
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
#data=data[:,[0,4]]


#Standardize data (substract mean divide with std)
data= (data - np.mean(data)) / np.std(data)



Tdata = PolyTransform(data,degree = 2)

#print(Tdata)
#quit()

# NOTE: So this transform does not help. Am I made a mistake (probably, how should i know).
# How is this should work? When was this coverd on the lecture?

print("Fit with regular data:")
train_data,test_data,train_labels,test_labels = train_test_split(data,labels,0.5)
clf = LogisticRegression(random_state=1,max_iter=1000).fit(train_data, train_labels)
print("In error:",clf.score(train_data, train_labels))
print("Out error:",clf.score(test_data, test_labels))

# So this is worse.... :(
print("Fit with Transformed data:")
train_data,test_data,train_labels,test_labels = train_test_split(Tdata,labels,0.5)
clf = LogisticRegression(random_state=1,max_iter=1000).fit(train_data, train_labels)
print("In error:",clf.score(train_data, train_labels))
print("Out error:",clf.score(test_data, test_labels))


#task 3
# NOTE: Please explaine me how is this task make any sense?
def visualize(data,labels,a,b):
	plt.scatter(data[:,0],data[:,1],c=labels)
	xx = np.linspace(min(data[:,0]) , max(data[:,0]))
	yy = a * xx + b
	plt.plot(xx, yy, 'k-')



from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

data=data[:,[0,4]]
Tdata = PolyTransform(data,degree = 3)
clf = LogisticRegression(random_state=1,max_iter=1000).fit(Tdata, labels)
#print(clf.)

w = clf.coef_
a = w[0,0] / w[0,1]
b =  3.5 #hack
visualize(data,labels,a,b)

#_, ax = plt.subplots(figsize=(4, 3))
#DecisionBoundaryDisplay.from_estimator(
#    clf,
#    Tdata,
#    cmap=plt.cm.Paired,
#    ax=ax,
#    response_method="predict",
#    plot_method="pcolormesh",
#    shading="auto",
#    eps=0.5,
#    n_features = 6,
#)

# Plot also the training points
#plt.scatter(Tdata[:, 0], Tdata[:, 1], c=labels, edgecolors="k", cmap=plt.cm.Paired)

#NOTE: I hate machine learning, and this is why.


plt.xticks(())
plt.yticks(())

plt.show()





quit()





