import math
import pandas as pd
import numpy as np
import numpy as np
from neupy import algorithms


iris_location = "data/iris.data"
wine_location = "data/wine.data"
wine_names = ['Class','Alcohol', 'Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue OD280/OD315 of diluted wines',
'Proline']
iris_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
iris_data = pd.read_csv(iris_location, names=iris_names, index_col=False)
wine_data = pd.read_csv(wine_location, names=wine_names, index_col=False)
wine_data = wine_data.drop(wine_data.columns[[0]], axis=1)
iris_data = iris_data.drop(iris_data.columns[[4]], axis=1)


def d(mas):
    df = pd.DataFrame(mas)
    df = df.transform(lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1)
    df = df.transform(lambda x: x - x.mean())
    mas = np.array(df)
    return mas


iris_data = d(iris_data)
wine_data = d(wine_data)

w_start = np.array(np.random.uniform(-1,1,size=(4,1)))

def norm(w):
    w = w/ (math.sqrt(sum(w*w)))
    return w


w_start = norm(w_start)

def matrixmult(m1, m2):
    s=0 #сумма
    t = [] # временная матрица
    m3 = [] # конечная матрица
    if len(m2) != len(m1[0]):
        print("Матрицы не могут быть перемножены")
    else:
        r1 = len(m1) # количество строк в первой матрице
        c1 = len(m1[0]) # Количество столбцов в 1
        r2 = c1 # и строк во 2ой матрице
        c2 = len(m2[0]) # количество столбцов во 2ой матрице
        for z in range(0, r1):
            for j in range(0, c2):
                for i in range(0, c1):
                    s = s + m1[z][i] * m2[i][j]
                t.append(s)
                s=0
            m3.append(t)
            t = []

    return m3


def trans(mas):
    trans = list()
    for i in range(len(mas[0])):
        l = list()
        for j in range(len(mas)):
            l.append(mas[j][i])
        print('')
        trans.append(l)
    return np.asarray(trans)



def oya(w_start, y_start, x, k):
    for j in range(10**k):
        for i in range (k):
            a = matrixmult(trans(y_start), trans(w_start))
            a = np.array(a)
            b = trans(x) - trans(a)
            b = np.array(b)
            el = np.array(k)
            c = y_start / el
            c = np.array(c)
            d = matrixmult(c, trans(b))
            d = np.array(d)
            w_start = trans(trans(w_start) - d)
            w_start = norm(w_start)
            w_start = np.array(w_start)
        y_start = matrixmult(trans(w_start),trans(x))
        y_start = np.array(y_start)
    return w_start, y_start, x








iris_data_t = trans(iris_data)
iris_data_t = np.array(iris_data_t)
w_start = np.array(w_start)
y_start1 = matrixmult(trans(w_start),iris_data_t)
y_start1 = np.array(y_start1)
a = oya(w_start,y_start1,iris_data,4)
a = np.array(a)
check = matrixmult(np.array(a[0]),np.array(a[1]))
check = np.array(check)
iris_data = np.array(iris_data)

check_t = trans(check)
print('final results(iris)')
print(check_t)
print('iris data')
print(iris_data)
print('decompressing(iris)')
print(check_t - iris_data)

iris_data = np.array(iris_data)
ojanet = algorithms.Oja(
         minimized_data_size=1,
         step=0.01,
         verbose=False
     )

ojanet.train(iris_data, epochs=100)
minimized = ojanet.predict(iris_data)
print('Minimized(iris, neupy)',minimized)
print('Reconstruct(iris)', ojanet.reconstruct(minimized))



wine_data = np.array(wine_data)




ojanet = algorithms.Oja(
         minimized_data_size=1,
         step=0.01,
         verbose=False
     )

ojanet.train(wine_data, epochs=100)
minimized = ojanet.predict(wine_data)
print('Minimized(wine, neupy)',minimized)
print('Reconstruct(wine)', ojanet.reconstruct(minimized))

