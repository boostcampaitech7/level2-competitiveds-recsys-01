import os
import time

from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils.constant_utils import Directory

def inference(model, type, X, y=None):
    if type=='validation':
        prediction = model.predict(X)
        mae = mean_absolute_error(y, prediction)
        return prediction, mae
    prediction = model.predict(X)
    sample_submission = Directory.sample_submission
    sample_submission['deposit'] = prediction

    return sample_submission