import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_digits, load_iris
from sklearn.model_selection import train_test_split #data is split into two data
from sklearn.naive_bayes import GaussianNB
iris= load_iris()
wine = load_wine()
digits = load_digits()

data = np.array(pd.read_csv('data/zoo.csv')) # read iris.csv into numpy array

#X = data[:,:-1] # predictor variables are the first four columns
#Y = data[:,-1:] # target variable is the last column
X = iris.data
# response vector in a object named y
Y = iris.target
# randomly splits the data into training and testing sets
X_train_5, X_test_5, Y_train_5, Y_test_5 = train_test_split(X, Y, test_size=.5) #50% is the train
X_train_6, X_test_6, Y_train_6, Y_test_6 = train_test_split(X, Y, test_size=.6) #60% is the train
X_train_7, X_test_7, Y_train_7, Y_test_7 = train_test_split(X, Y, test_size=.7) #70% is the train
X_train_8, X_test_8, Y_train_8, Y_test_8 = train_test_split(X, Y, test_size=.8) #80% is the train
X_train_9, X_test_9, Y_train_9, Y_test_9 = train_test_split(X, Y, test_size=.9) #90% is the train

# create a 5-NN classifier
model1 = GaussianNB() #use 5 nearest neighbors
model2 = GaussianNB() #use 5 nearest neighbors
model3 = GaussianNB() #use 5 nearest neighbors
model4 = GaussianNB() #use 5 nearest neighbors
model5 = GaussianNB() #use 5 nearest neighbors


# fit the model to the training data
model1.fit(X_train_5, Y_train_5)
model2.fit(X_train_6, Y_train_6)
model3.fit(X_train_7, Y_train_7)
model4.fit(X_train_8, Y_train_8)
model5.fit(X_train_9, Y_train_9)

# evaluate with the testing data
print("testing accuracy Iris(train/test - 50/50): ", model1.score(X_test_5, Y_test_5) * 100, "%", sep = "")
print("testing accuracy Iris(train/test - 60/40): ", model2.score(X_test_6, Y_test_6) * 100, "%", sep = "")
print("testing accuracy Iris(train/test - 70/30): ", model3.score(X_test_7, Y_test_7) * 100, "%", sep = "")
print("testing accuracy Iris(train/test - 80/20): ", model4.score(X_test_8, Y_test_8) * 100, "%", sep = "")
print("testing accuracy Iris(train/test - 90/10): ", model5.score(X_test_9, Y_test_9) * 100, "%", sep = "")
import matplotlib.pyplot as plt
x = [model1.score(X_test_5, Y_test_5) * 100, model2.score(X_test_6, Y_test_6) * 100,model3.score(X_test_7, Y_test_7) * 100,model4.score(X_test_8, Y_test_8) * 100,model5.score(X_test_9, Y_test_9) * 100]
z1 = [50 , 60, 70, 80, 90]
fig = plt.figure()
plt.errorbar(z1, x,  xerr=1, yerr=0.5)
plt.title('Simple error bar chart')
plt.grid(True)

# Setting the background color
plt.xlabel('Процент обучающей выборки %')
plt.ylabel('Точность %')
plt.title('Iris')
ax = plt.axes()
ax.set_facecolor("yellow")
plt.show()
# predict the class of a new iris
# new_iris = [5.2, 3.4, 1.3, .2] # probably setosa (class 0)
# print("new iris probably belongs to class", model.predict([new_iris]))
# print("class probabilities:", kNN.predict_proba([new_iris]))

#might need to refresh with each run...?
