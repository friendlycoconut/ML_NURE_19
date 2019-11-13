import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# загружаем данные для обработки (они хранятся в датасетах scikit-learn)
iris = datasets.load_iris()
X = iris.data[:, :2]  # нам нужны только 2 первых колонки
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()


plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Длина чашелистника')
plt.ylabel('Ширина чашелистника')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# Чтобы лучше понять взаимодействие измерений, постройте первые три измерения PCA
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("Первые три измерения PCA")
ax.set_xlabel("1-ое измерение")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2-ое измерение")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3-е измерение")
ax.w_zaxis.set_ticklabels([])

plt.figure(figsize=(10, 7))

features = iris.data.T
petal_length = features[3]
x= petal_length
plt.hist(x, bins=20, color="green")
plt.title("Petal Length in cm")
plt.xlabel("Petal_Length_cm")
plt.ylabel("Count")


plt.show()

