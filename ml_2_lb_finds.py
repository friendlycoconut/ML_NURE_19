import csv

import pandas
import seaborn
import numpy as np
import sys

from matplotlib.pyplot import gcf
from networkx.drawing.tests.test_pylab import plt
from pandas import read_csv
from pandas import set_option
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from subprocess import check_output




def instancesNum(DataFrame):
    num_of_instances = 1
    num_of_attr = len(DataFrame.columns)
    for i in range(1, num_of_attr):
        num_of_instances *= (DataFrame[DataFrame.columns[i]].nunique())
    return num_of_instances


def hypothesisNum(DataFrame):
    num_of_hypothesis = 1
    num_of_attr = len(DataFrame.columns)
    for i in range(1, num_of_attr):
        num_of_hypothesis *= (DataFrame[DataFrame.columns[i]].nunique() + 1)
    return num_of_hypothesis + 1


# load file [class.csv] into dataframe [_df_class]
_df_c = read_csv('data/class.csv')
# load file [zoo.csv] into dataframe [_df_zoo]
_df_z = read_csv('data/zoo.csv')

print(_df_z.dtypes.value_counts())

print("Number of instances(without individual 'animal_name' field): ", instancesNum(_df_z))
print("Number of hypothesis(without individual 'animal_name' field): ", hypothesisNum(_df_z))


##############################
def find_s(instances):
    features, labels = instances
    if len(features) == 0:
        return []
    hypothesis = [None] * len(features[0])
    for i in range(0, len(features)):
        if labels[i]==1:
            attributes = features[i]
            for j in range(0, len(attributes)):
                if hypothesis[j] == None:
                    hypothesis[j] = attributes[j]
                elif hypothesis[j] != attributes[j]:
                    hypothesis[j] = "?"
    return hypothesis
def candidate_elimination(concepts, target):

    specific_h = concepts[0].copy()
    print("initialization of specific_h and general_h")
    print(specific_h)

    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    print(general_h)

    for i, h in enumerate(concepts):

        if target[i] == 1:
            for x in range(len(specific_h)):

                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'

        # Checking if the hypothesis has a positive target
        if target[i] != 1:
            for x in range(len(specific_h)):

                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
        print(" steps of Candidate Elimination Algorithm",i+1)
        print(specific_h)
        print(general_h)

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?']]
    for i in indices:

        general_h.remove(['?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?'])

    return specific_h, general_h

def build_hypothesis_space(G, S):
    list_of_hypotheses = []
    k = S.tolist()

    for list_of_Gs in G:
        for x in list_of_Gs:
            if x != '?':
                index_of_x_el = list_of_Gs.index(x)

                for y in range(len(k)):
                    if k[y] != '?':
                        index_of_y_el = y
                        if(index_of_y_el!=index_of_x_el):
                            sub_list = []
                            for number in range(1, 18):
                                sub_list.append('?')
                            if(sub_list[index_of_y_el]  == '?'):
                                sub_list[index_of_x_el] = x
                                sub_list[index_of_y_el] = k[y]
                                list_of_hypotheses.append(sub_list)

    return list_of_hypotheses


for i in range (1,8):
    hypothesis = find_s([_df_z[['animal_name', 'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator',
                            'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic',
                            'catsize']].values,
                     _df_z['class_type'].apply(lambda x: 1 if x == i else 0).values])
    print("Number of class: ",i ," ", hypothesis)

# Separating concept features from Target
concepts = np.array(_df_z.iloc[:,0:-1])

# Isolating target into a separate DataFrame
#copying last column to target array
target = np.array(_df_z.iloc[:,-1])
s_final, g_final = candidate_elimination(concepts, target)
print("Final Specific_h:", s_final)
print("Final General_h:", g_final)
print("Restoring hypothesis space ",build_hypothesis_space(g_final,s_final))