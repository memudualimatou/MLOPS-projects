# Importing dataset

import warnings

warnings.filterwarnings('ignore')

# importing the libraries
import numpy as np
import pandas as pd
import os
import warnings
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow.sklearn
import mlflow

# importing dataset
wine_data = pd.read_csv('C:\MINE\DATA SCIENCE\my datasets\winequality-red.csv')
wine_data.head()

# checking shape
print("wine dataset shape:", wine_data.shape)

# visualizing features info

print(wine_data.describe().T)

# checking columns datatypes

print(wine_data.info())

# splitting data

X = wine_data.drop(columns='quality')
y = wine_data[['quality']]
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)


# metrics function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# Defining model parameters
alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

# Running MLFlow script
with mlflow.start_run():
    # Instantiating model with model parameters
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)

    # Fitting training data to the model
    model.fit(X_train, y_train)

    # Running prediction on validation dataset
    preds = model.predict(X_val)

    # Getting metrics on the validation dataset
    (rmse, mae, r2) = eval_metrics(y_val, preds)
    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # Logging params and metrics to MLFlow
    mlflow.log_param('alpha', alpha)
    mlflow.log_param('l1_ratio', l1_ratio)
    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('abs_error', mae)
    mlflow.log_metric('r2', r2)

    # Logging model to MLFlow
    mlflow.sklearn.log_model(model, 'model')
