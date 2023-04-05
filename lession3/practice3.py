import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

import matplotlib.pyplot as plt

import random

def train_test_split(X,Y,p):
	train_x = []
	train_y = []
	seq = range(len(X))
	idx = random.sample(seq, int(p*len(X)))
	test_idx = np.setdiff1d(seq, idx)
	train_x = X[idx]
	train_y = Y[idx]
	test_x = X[test_idx]
	test_y = Y[test_idx]
	return train_x,train_y,test_x,test_y










data=pd.read_csv('./data/poly.csv')


X=data.values[:,0]
Y=data.values[:,1]


plt.figure("pontok")

plt.scatter(X,Y)


for p in range(1,11):
	poly = np.polynomial.polynomial.Polynomial.fit(X,Y,p)
	x = np.linspace(min(X),max(X),100)
	y = poly(x)
	plt.plot(x,y,label = p)

plt.legend()
train_x,train_y,test_x,test_y = train_test_split(X,Y,0.5)

#print(len(train_x))
#print(len(X))

out_error = []
in_error = []

ps = range(30)
min_error = 1000000000
min_p = -1
	
for p in ps:
	poly = np.polynomial.polynomial.Polynomial.fit(train_x,train_y,p)

	predict = poly(test_x)
	mse = (np.square(predict - test_y)).mean()
	out_error.append(mse)
	if min_error > mse:
		min_error = mse
		min_p = p
	predict = poly(train_x)
	mse = (np.square(predict - train_y)).mean()
	in_error.append(mse)

print("Best polinom is degree:",min_p)

plt.figure("error")
plt.plot(ps,out_error,label = "Out error")
plt.plot(ps,in_error,label = "In error")
plt.legend()


#print(y)
plt.show()
#print(data)