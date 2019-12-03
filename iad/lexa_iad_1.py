import numpy as np
import pandas as pd
import math
from sklearn.datasets import load_iris
data = load_iris()
iris_data = data.data
attributes = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'CLASS']
def column_arr(arr):
   col_array = []
   for i in range(len(arr)):
       for j in range(len(arr[i])):
           if j in range(-len(col_array), len(col_array)):
               col_array[j].append(arr[i][j])
           else:
               col_array.insert(j, [arr[i][j]])
   return col_array
def avg(arr):
   summ = 0
   for item in arr:
       summ += item
   return round(summ / len(arr), 6)
def dataFrame(arr, target):
   iris = {}
   c_array = column_arr(arr)
   c_array.append(target)
   for item in range(len(attributes)):
       iris[attributes[item]] = c_array[item][:]
   df = pd.DataFrame(iris, columns=attributes)
   datasets = {}
   by_class = df.groupby('CLASS')
   for groups, data in by_class:
       datasets[groups] = data
   return datasets
def attribute_avg(arr, target):
   datasets = dataFrame(arr, target)
   avg_arr = []
   for i in range(len(datasets)):
       datasets[i] = datasets[i].drop('CLASS', 1)
       column_avg = []
       for column in datasets[i]:
           columnSeriesObj = datasets[i][column]
           column_avg.append(avg(columnSeriesObj.values))
       avg_arr.append(column_avg)
   return avg_arr
print(attribute_avg(iris_data, data.target))
