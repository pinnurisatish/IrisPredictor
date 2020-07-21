import pandas as pd 
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from  sklearn import  datasets
import pickle
data=pd.read_csv('Iris.csv')
# iris=datasets.load_iris()
# x=iris.data
# y=iris.target
#print(y)
# x=data.iloc[:,:4]
# y=data['variety']
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25)
svc=SVC()
svc.fit(x_train,y_train)
import pickle
pickle.dump(svc, open('model.pkl', 'wb'))

