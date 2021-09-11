# import tensorflow
# from tensorflow import keras
import pandas as pd
import numpy as np
import sklearn


from sklearn import linear_model

import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

#Data cleansing, taking out the required columns for the outcome and aligning it
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1","G2","G3","studytime","failures","absences"]]


#We are going to predict G3 wrt to other attributes
predict = "G3"

X = np.array(data.drop([predict],1))
Y= np.array(data[predict])

#Training and testing data seperation
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.1)

'''
best = 0
for _ in range(30):
    #Training and testing data seperation
    x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.1)

    #Training or fitting the model for the data given with Linear_Regression Algorithm
    linear = linear_model.LinearRegression()
    linear.fit(x_train,y_train)

    #checking the accuracy #Saves model, Creating pickle file
    accuracy = linear.score(x_test,y_test)
    print(accuracy)
    print("co-efficient:", linear.coef_)
    print("Intercept:", linear.intercept_)

    if accuracy>best:
        best = accuracy
        with open("studentmodel.pickle","wb") as f:
            pickle.dump(linear,f)'''

#read pickle file created with saved model
picke_in = open("studentmodel.pickle","rb")
linear = pickle.load(picke_in)

#Final part of code...Predicting the result required
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x],y_train[x],x_test[x],y_test[x])

p = 'G2'
style.use("ggplot")
pyplot.scatter(data[p],data['G3'])
pyplot.xlabel(p)
pyplot.ylabel('Final Grade')
# pyplot.show()

