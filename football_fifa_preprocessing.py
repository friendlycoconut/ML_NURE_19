
import pandas as pd
from sklearn.model_selection import train_test_split

dataset_url = 'data/football.csv'
data = pd.read_csv(dataset_url)
print (data.head())
y = data.Position
X = data.drop('ID', axis=1)
print(X)
X = X.drop('Name', axis=1)
X = X.drop('Name', axis=1)
X = X.drop('Name', axis=1)
X = X.drop('Name', axis=1)
X = X.drop('Name', axis=1)
X = X.drop('Name', axis=1)
X = X.drop('Name', axis=1)
X = X.drop('Name', axis=1)

print(X)


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
print("\nX_train:\n")
print(X_train.head())
print(X_train.shape)

print("\nX_test:\n")
print(X_test.head())
print(X_test.shape)
X_train.to_csv('data/train_set_football.csv', index=False)
X_test.to_csv('data/test_set_football.csv', index=False)