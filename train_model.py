import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump

# load dataset
digits = datasets.load_digits()
# 80/20 split between training and testing data
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.8, random_state=42)
# create instance of model
knn = KNeighborsClassifier(n_neighbors=7)
# fit model
knn.fit(x_train, y_train)
# print accuracy
print(knn.score(x_test, y_test))
# dump into model.joblib file
dump(knn, "model.joblib")