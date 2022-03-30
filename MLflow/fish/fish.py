# Importing dataset

import warnings

warnings.filterwarnings('ignore')

# importing the libraries
import numpy as np
import pandas as pd
import os
import warnings
import sys
from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from urllib.parse import urlparse
import mlflow.sklearn
import mlflow

# importing dataset
fish_data = pd.read_csv('C:\MINE\DATA SCIENCE\my datasets\Fish2.csv')
fish_data.head()

# checking shape
print("fish dataset shape:", fish_data.shape)

# visualizing features info
print(fish_data.describe().T)

# checking columns datatypes
print(fish_data.info())

# spliting date

X = fish_data.iloc[:, :-1].values
y = fish_data.iloc[:, 6].values

# encoding target data
y_le = LabelEncoder()
y = y_le.fit_transform(y)

# splitting data
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)


# metrics function
def eval_metrics(actual, pred):
    acc = accuracy_score(actual, pred)
    matrix = confusion_matrix(actual, pred)
    return acc, matrix


# Defining model parameters
random_state = int(sys.argv[1]) if len(sys.argv) > 150 else 42
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
penalty = str(sys.argv[3]) if len(sys.argv) == 0 else 'l2'

# Running MLFlow script
with mlflow.start_run():
    # Instantiating model with model parameters
    model = LogisticRegression(random_state=random_state, l1_ratio=l1_ratio, penalty=penalty)

    # Fitting training data to the model
    model.fit(X_train, y_train)

    # Running prediction on validation dataset
    preds = model.predict(X_val)

    # Getting metrics on the validation dataset
    (acc, matrix) = eval_metrics(y_val, preds)
    print(f"Elasticnet model (random_state=%f, l1_ratio=%f,penalty={penalty}):" % (random_state, l1_ratio), format(penalty))
    print("  accuracy: %s" % acc)
    print("  confusion matrix: %s" % matrix)

    # Logging params and metrics to MLFlow
    mlflow.log_param('random state', random_state)
    mlflow.log_param('l1_ratio', l1_ratio)
    mlflow.log_param('penalty', penalty)
    mlflow.log_metric('accuracy', acc)

    # Logging model to MLFlow
    mlflow.sklearn.log_model(model, 'model')
