"""
Make the imports of python packages needed
"""
import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.datasets import load_iris


# Import the dataset and define the feature as well as the target datasets / columns#
from sklearn.tree import DecisionTreeClassifier

dataset_zoo = pd.read_csv('data/zoo.csv',
                      names=['animal_name','hair', 'feathers', 'eggs', 'milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize','class_type'])  # Import all columns omitting the fist which consists the names of the animals
dataset_iris = pd.read_csv('data/iris.csv',
                           names=['sepal.length','sepal.width','petal.length','petal.width','species'])
# We drop the animal names since this is not a good feature to split the data on
dataset_zoo = dataset_zoo.drop('animal_name', axis=1)


###################
def entropy(target_col):

    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum(
        [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy


def InfoGain(data, split_attribute_name, target_name):

    total_entropy = entropy(data[target_name])


    vals, counts = np.unique(data[split_attribute_name], return_counts=True)


    Weighted_Entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])


    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain



def ID3(data, originaldata, features, target_attribute_name, parent_node_class=None):

    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    # If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[
            np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]

    # If the feature space is empty, return the mode target feature value of the direct parent node --> Note that
    # the direct parent node is that node which has called the current run of the ID3 algorithm and hence
    # the mode target feature value is stored in the parent_node_class variable.

    elif len(features) == 0:
        return parent_node_class

    # If none of the above holds true, grow the tree!

    else:
        # Set the default value for this node --> The mode target feature value of the current node
        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

        # Select the feature which best splits the dataset
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in
                       features]  # Return the information gain values for the features in the dataset
       
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        # Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
        # gain in the first run
        tree = {best_feature: {}}

        # Remove the feature with the best inforamtion gain from the feature space
        features = [i for i in features if i != best_feature]

        # Grow a branch under the root node for each possible value of the root node feature

        for value in np.unique(data[best_feature]):
            value = value
            # Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
            sub_data = data.where(data[best_feature] == value).dropna()

            # Call the ID3 algorithm for each of those sub_datasets with the new parameters --> Here the recursion comes in!
            subtree = ID3(sub_data, originaldata, features, target_attribute_name, parent_node_class)

            # Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree

        return (tree)

    ###################


###################


def predict(query, tree, default=1):

    for key in list(query.keys()):
        if key in list(tree.keys()):
            # 2.
            try:
                result = tree[key][query[key]]
            except:
                return default

            # 3.
            result = tree[key][query[key]]
            # 4.
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result


"""
Check the accuracy of our prediction.
The train_test_split function takes the dataset as parameter which should be divided into
a training and a testing set. The test function takes two parameters, which are the testing data as well as the tree model.
"""


###################
###################
def train_test_split(dataset):
    training_data = dataset.iloc[:70].reset_index(drop=True)  # We drop the index respectively relabel the index
    # starting form 0, because we do not want to run into errors regarding the row labels / indexes
    testing_data = dataset.iloc[70:].reset_index(drop=True)
    return training_data, testing_data

def train_test_split_iris(dataset):
    training_data = dataset.iloc[:105].reset_index(drop=True)  # We drop the index respectively relabel the index
    # starting form 0, because we do not want to run into errors regarding the row labels / indexes
    testing_data = dataset.iloc[105:].reset_index(drop=True)
    return training_data, testing_data


training_data_zoo = train_test_split(dataset_zoo)[0]
testing_data_zoo = train_test_split(dataset_zoo)[1]

training_data_iris = train_test_split_iris(dataset_iris)[0]
testing_data_iris =  train_test_split_iris(dataset_iris)[1]


def test(data, tree,target_attr):
    # Create new query instances by simply removing the target feature column from the original dataset and
    # convert it to a dictionary
    queries = data.iloc[:, :-1].to_dict(orient="records")

    # Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"])

    # Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i, "predicted"] = predict(queries[i], tree, 1)
    print('The prediction accuracy is: ', (np.sum(predicted["predicted"] == data[target_attr]) / len(data)) * 100, '%')


"""
Train the tree, Print the tree and predict the accuracy
"""
tree_zoo = ID3(training_data_zoo, training_data_zoo, training_data_zoo.columns[:-1],'class_type')
tree_iris_1 = ID3(training_data_iris, training_data_iris, dataset_iris.columns[:-1],'species')
tree_iris_2 = ID3(dataset_iris, dataset_iris, dataset_iris.columns[:-1],'species')
print("zoo tree")
pprint(tree_zoo)
print("iris tree - 1")
pprint(tree_iris_1)
print("iris tree - 2")
pprint(tree_iris_2)
print("accuracy for zoo")
test(testing_data_zoo, tree_zoo,'class_type')
print("accuracy for iris_1")
test(testing_data_iris, tree_iris_1,'species')
print("accuracy for iris_2")
test(testing_data_iris, tree_iris_2,'species')

data = pd.read_csv('data/iris.csv')
data.info()
data.isnull().sum()
data.describe()

from sklearn import model_selection
target = data['species']
data.drop('species', axis=1, inplace=True)
x_train, x_test, y_train, y_test = model_selection.train_test_split(data, target, random_state=0)
x_train.head()

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)
y_pred = dtree.predict(x_test)

dtree.score(x_test, y_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predictions'])

dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=20)
dt.fit(x_train, y_train)
y_p = dt.predict(x_test)
pd.crosstab(y_test, y_p, rownames=['Actual'], colnames=['Predicted'])
