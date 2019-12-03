import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def matrixmult(m1, m2):
    s = 0  # сумма
    t = []  # временная матрица
    m3 = []  # конечная матрица
    if len(m2) != len(m1[0]):
        print("Матрицы не могут быть перемножены")
    else:
        r1 = len(m1)  # количество строк в первой матрице
        c1 = len(m1[0])  # Количество столбцов в 1
        r2 = c1  # и строк во 2ой матрице
        c2 = len(m2[0])  # количество столбцов во 2ой матрице
        for z in range(0, r1):
            for j in range(0, c2):
                for i in range(0, c1):
                    s = s + m1[z][i] * m2[i][j]
                t.append(s)
                s = 0
            m3.append(t)
            t = []
    return m3


def transponir(mas):
    trans = list()
    for i in range(len(mas[0])):
        l = list()
        for j in range(len(mas)):
            l.append(mas[j][i])
        print('')
        trans.append(l)
    return np.asarray(trans)


def pick_nonzero_row(m, k):
    while k < m.shape[0] and not m[k, k]:
        k += 1
    return k


def very_feel_func(x, y):
    t = transponir(x)
    first = matrixmult(t, x)
    first = np.array(first)
    pr = matrixmult(t, y)
    first1 = np.matrix(first)

    m = np.hstack((first1,
                   np.matrix(np.diag([1.0 for i in range(first1.shape[0])]))))

    # forward trace
    for k in range(len(first1)):
        swap_row = pick_nonzero_row(m, k)
        if swap_row != k:
            m[k, :], m[swap_row, :] = m[swap_row, :], np.copy(m[k, :])
        if m[k, k] != 1:
            m[k, :] *= 1 / m[k, k]
        for row in range(k + 1, len(first1)):
            m[row, :] -= m[k, :] * m[row, k]

    # backward trace
    for k in range(len(first1) - 1, 0, -1):
        for row in range(k - 1, -1, -1):
            if m[row, k]:
                m[row, :] -= m[k, :] * m[row, k]
    obratn = np.hsplit(m, len(first1) // 2)[1]

    pr1 = np.array(pr)
    obratn1 = np.array(obratn)

    a = matrixmult(obratn1, pr1)
    return a


colnames = ['V', 'F', 'C', 'M', 'Column5']
datContent = [i.strip().split() for i in open("data/Data.dat").readlines()]
data = pd.DataFrame(datContent, columns=colnames)
print(data)
del data['Column5']
idx = 0
data.insert(0, 'X0', 1)
print("--")
print(data)

cols = data.columns.tolist()

data = data[cols]
d = np.array(data)
xx1 = np.zeros((59, 4))
xx2 = np.zeros((59, 4))
xx3 = np.zeros((59, 4))
xx4 = np.zeros((59, 4))
xx5 = np.zeros((59, 4))
xx6 = np.zeros((59, 4))
yy1 = np.zeros((59, 1))
yy2 = np.zeros((59, 1))
yy3 = np.zeros((59, 1))
yy4 = np.zeros((59, 1))
yy5 = np.zeros((59, 1))
yy6 = np.zeros((59, 1))
for i in range(59):
    xx1[i, 0] = d[i + 1, 0]  # 1
    xx1[i, 1] = d[i, 2]  # 1
    xx1[i, 2] = d[i + 1, 4]  # 1
    xx1[i, 3] = d[i, 1]  # 1
    yy1[i, 0] = d[i + 1, 3]  # 1

    xx2[i, 0] = d[i + 1, 0]  # 2
    xx2[i, 1] = d[i, 4]  # 2
    xx2[i, 2] = d[i, 2]  # 2
    xx2[i, 3] = d[i + 1, 3]  # 2
    yy2[i, 0] = d[i + 1, 1]  # 2

    xx3[i, 0] = d[i + 1, 0]  # 3
    xx3[i, 1] = d[i, 3]  # 3
    xx3[i, 2] = d[i + 1, 1]  # 3
    xx3[i, 3] = d[i + 1, 2]  # 3
    yy3[i, 0] = d[i + 1, 4]  # 3

    xx4[i, 0] = d[i + 1, 0]  # 4
    xx4[i, 1] = d[i, 4]  # 4
    xx4[i, 2] = d[i + 1, 1]  # 4
    xx4[i, 3] = d[i, 3]  # 4
    yy4[i, 0] = d[i + 1, 2]  # 4

    xx5[i, 0] = d[i + 1, 0]  # 5
    xx5[i, 1] = d[i, 4]  # 5
    xx5[i, 2] = d[i + 1, 2]  # 5
    xx5[i, 3] = d[i + 1, 1]  # 5
    yy5[i, 0] = d[i + 1, 3]  # 5

    xx6[i, 0] = d[i + 1, 0]  # 6
    xx6[i, 1] = d[i, 1]  # 6
    xx6[i, 2] = d[i + 1, 3]  # 6
    xx6[i, 3] = d[i + 1, 4]  # 6
    yy6[i, 0] = d[i + 1, 2]  # 6

x1 = np.array(xx1)
x2 = np.array(xx2)
x3 = np.array(xx3)
x4 = np.array(xx4)
x5 = np.array(xx5)
x6 = np.array(xx6)

y1 = np.array(yy1)
y2 = np.array(yy2)
y3 = np.array(yy3)
y4 = np.array(yy4)
y5 = np.array(yy5)
y6 = np.array(yy6)

a1 = very_feel_func(x1, y1)
a2 = very_feel_func(x2, y2)
a3 = very_feel_func(x3, y3)
a4 = very_feel_func(x4, y4)
a5 = very_feel_func(x5, y5)
a6 = very_feel_func(x6, y6)

print("Coefficients(with (k-1)) 1 example: ", a1)
print("Coefficients(with (k-1)) 2 example: ", a2)
print("Coefficients(with (k-1)) 3 example: ", a3)
print("Coefficients(with (k-1)) 4 example: ", a4)
print("Coefficients(with (k-1)) 5 example: ", a5)
print("Coefficients(with (k-1)) 6 example: ", a6)

print('////////////////////////////////////////////////////////////////////////////////')
X1 = data[['X0', 'F', 'M', 'V']].values
Y1 = data['C'].values
X2 = data[['X0', 'M', 'F', 'C']].values
Y2 = data['V'].values
X3 = data[['X0', 'C', 'V', 'F']].values
Y3 = data['M'].values
X4 = data[['X0', 'M', 'V', 'C']].values
Y4 = data['F'].values
X5 = data[['X0', 'M', 'F', 'V']].values
Y5 = data['C'].values
X6 = data[['X0', 'V', 'C', 'M']].values
Y6 = data['F'].values

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size=0.2, random_state=0)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, Y2, test_size=0.2, random_state=0)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, Y3, test_size=0.2, random_state=0)
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, Y4, test_size=0.2, random_state=0)
X5_train, X5_test, y5_train, y5_test = train_test_split(X5, Y5, test_size=0.2, random_state=0)
X6_train, X6_test, y6_train, y6_test = train_test_split(X6, Y6, test_size=0.2, random_state=0)


regr1 = linear_model.LinearRegression()
regr1.fit(X1_train, y1_train)

print('Intercept (1 example): \n', regr1.intercept_)
print('Coefficients(without (k-1)): \n', regr1.coef_)
y_pred = regr1.predict(X1_test)
df = pd.DataFrame({'Actual': y1_test, 'Predicted': y_pred})

print('R2 score (1 example):',r2_score(y1_test, y_pred))

regr2 = linear_model.LinearRegression()
regr2.fit(X2_train, y2_train)

print('Intercept (2 example): \n', regr2.intercept_)
print('Coefficients(without (k-1)): \n', regr2.coef_)
y_pred = regr2.predict(X2_test)
df = pd.DataFrame({'Actual': y2_test, 'Predicted': y_pred})

print('R2 score (2 example):',r2_score(y2_test, y_pred))