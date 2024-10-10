import os
import time

from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils.constant_utils import Directory

def inference(X, y, model):
    prediction = model.predict(X)
    mae = mean_absolute_error(y, prediction)

    return prediction, mae

