from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


wine_location = "data/wine.data"
wine_data = pd.read_csv(wine_location)
df = pd.DataFrame(wine_data)
df = df.transform(lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1)
df = df.sample(frac=1)
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
df = np.array(df)
k = 3
C_x1 = np.array(np.random.uniform(-1,1,size=k))
C_x2 = np.array(np.random.uniform(-1,1,size=k))
C_x3 = np.array(np.random.uniform(-1,1,size=k))
C_x4 = np.array(np.random.uniform(-1,1,size=k))
C_x5 = np.array(np.random.uniform(-1,1,size=k))
C_x6 = np.array(np.random.uniform(-1,1,size=k))
C_x7 = np.array(np.random.uniform(-1,1,size=k))
C_x8 = np.array(np.random.uniform(-1,1,size=k))
C_x9 = np.array(np.random.uniform(-1,1,size=k))
C_x10 = np.array(np.random.uniform(-1,1,size=k))
C_x11 = np.array(np.random.uniform(-1,1,size=k))
C_x12 = np.array(np.random.uniform(-1,1,size=k))
C_x13 = np.array(np.random.uniform(-1,1,size=k))
C = np.array(list(zip(C_x1, C_x2, C_x3, C_x4,C_x5,C_x6,C_x7,C_x8,C_x9,C_x10,C_x11,C_x12,C_x13)), dtype=np.float32)
f1 = df[:,0]
f2 = df[:, 1]
f3 = df[:, 2]
f4 = df[:, 3]
f5 = df[:,4]
f6 = df[:, 5]
f7 = df[:, 6]
f8 = df[:, 7]
f9 = df[:,8]
f10 = df[:, 9]
f11 = df[:, 10]
f12 = df[:, 11]
f13 = df[:, 12]
X = np.array(list(zip(f1, f2,f3,f4,f5, f6,f7,f8,f9, f10,f11,f12,f13)))
C_old = np.zeros(C.shape)
clusters = np.zeros(len(X))
clusters1 = np.zeros(len(X))
error = dist(C, C_old, None)
while error != 0:
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    C_old = deepcopy(C)
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)
a = list()
b = list()
c = list()
for i in range(len(X)):
    distances1 = dist(X[i], C)
    cluster1 = np.argmin(distances1)
    clusters1[i] = cluster1
    if (clusters1[i] == 0):
        a.append(distances1[0])
    if (clusters1[i] == 1):
        b.append(distances1[1])
    if (clusters1[i] == 2):
        c.append(distances1[2])

m = list()
m.append(a)
m.append(b)
m.append(c)
matrix = np.array(m)
a = np.array(a)
s_a = sum(a)
b= np.array(b)
s_b = sum(b)
c= np.array(c)
s_c = sum(c)
v = list()
v.append(s_a)
v.append(s_b)
v.append(s_c)



print('Regular algorithm')
print('Центроиды')
print(C)
print('вектор сумм расстояний объектов кластера до его центроида')
print(v)
print('матрица расстояний объектов кластера до его центроида')
print(matrix)

# Number of clusters
kmeans = KMeans(n_clusters=3)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_


# Comparing with scikit-learn centroids
print('Scikit')
print('Центроиды')
print(centroids) # From sci-kit learn
