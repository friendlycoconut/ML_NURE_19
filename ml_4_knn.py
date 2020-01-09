from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits

from sklearn.preprocessing import MinMaxScaler



#import metrics model to check the accuracy
from sklearn import metrics

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
wine = load_wine()
digits = load_digits()

print(len(iris.data))
print(len(wine.data))
print(len(digits.data))
print(digits.data)
print(iris.data)
print(iris.feature_names)


from sklearn.preprocessing import StandardScaler
print(iris.target_names)

# Feature matrix in a object named X
X = wine.data
# response vector in a object named y
y = wine.target
def MMscaler(X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train[0])
    for i in range(0, len(X_train)):
        X_train[i] = scaler.transform(X_train[i])
        X_test[i] = scaler.transform(X_test[i])
    return X_train,X_test

def Zscaler(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train[0])
    for i in range(0, len(X_train)):
        X_train[i] = scaler.transform(X_train[i])
        X_test[i] = scaler.transform(X_test[i])
    return X_train,X_test



from sklearn.model_selection import train_test_split
X_train = []
X_test = []
y_train = []
y_test = []
for i in range(10, 100, 10):
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size= i / 100.0,random_state=4)
    X_train.append(X_tr)
    X_test.append(X_te)
    y_train.append(y_tr)
    y_test.append(y_te)

X_train,X_test = MMscaler(X_train,X_test)

scores_list1 = []
for (X_tr, X_te, y_tr, y_te) in zip(X_train, X_test, y_train, y_test):
        knn = KNeighborsClassifier(n_neighbors=1,weights  = "distance")
        knn.fit(X_tr,y_tr)
        y_pred=knn.predict(X_te)
        scores_list1.append(metrics.accuracy_score(y_te,y_pred))


#X_train,X_test = Zscaler(X_train,X_test)

scores_list3 = []
for X_tr, X_te, y_tr, y_te in zip(X_train, X_test, y_train, y_test):
        knn = KNeighborsClassifier(n_neighbors=3,weights  = "distance")
        knn.fit(X_tr,y_tr)
        y_pred=knn.predict(X_te)
        scores_list3.append(metrics.accuracy_score(y_te,y_pred))





scores_list5 = []
for X_tr, X_te, y_tr, y_te in zip(X_train, X_test, y_train, y_test):
        knn = KNeighborsClassifier(n_neighbors=5,weights  = "distance")
        knn.fit(X_tr,y_tr)
        y_pred=knn.predict(X_te)
        scores_list5.append(metrics.accuracy_score(y_te,y_pred))

scores7 = {}
scores_list7 = []
for X_tr, X_te, y_tr, y_te in zip(X_train, X_test, y_train, y_test):
        knn = KNeighborsClassifier(n_neighbors=7,weights  = "distance")
        knn.fit(X_tr,y_tr)
        y_pred=knn.predict(X_te)
        scores_list7.append(metrics.accuracy_score(y_te,y_pred))
#plot the relationship between K and the testing accuracy
plt.plot(range(10, 100, 10),scores_list1, label = "k = 1")
plt.plot(range(10, 100, 10),scores_list3, "--", label = "k = 3")
plt.plot(range(10, 100, 10),scores_list5, ":", label = "k = 5")
plt.plot(range(10, 100, 10),scores_list7, "-.", label = "k = 7")
plt.xlabel('Процент обучающей выборки %')
plt.ylabel('Точность')
plt.title('Wine, взвешенный, нормировка по размаху')
plt.legend(loc='best')
ax = plt.axes()
# Setting the background color
ax.set_facecolor("yellow")
plt.show()