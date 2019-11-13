import pandas as pd

def get_setosa():
    data = read_data()
    data['class'] = data['class'].replace('Iris-setosa', '1')
    data['class'] = data['class'].replace('Iris-versicolor', '-1')
    data['class'] = data['class'].replace('Iris-virginica', '-1')

    return data.to_numpy()

def get_versicolor():
    data = read_data()
    data['class'] = data['class'].replace('Iris-setosa', '-1')
    data['class'] = data['class'].replace('Iris-versicolor', '1')
    data['class'] = data['class'].replace('Iris-virginica', '-1')

    return data.to_numpy()

def get_virginica():
    data = read_data()
    data['class'] = data['class'].replace('Iris-setosa', '-1')
    data['class'] = data['class'].replace('Iris-versicolor', '-1')
    data['class'] = data['class'].replace('Iris-virginica', '1')

    return data.to_numpy()

def read_data():
    data = pd.read_csv("datasets/iris.data", header=None)
    data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

    return data