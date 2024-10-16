import os
import time

from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils.constant_utils import Directory

def inference(model, type, X, y=None):
    if type=='train':
        prediction = model.predict(X)
        mae = mean_absolute_error(y, prediction)
        print("Train Mae score")
        print(mae)
        return prediction, mae
    
    elif type=='validation':
        prediction = model.predict(X)
        mae = mean_absolute_error(y, prediction)
        print("validation Mae score")
        print(mae)
        return prediction, mae
    else:
        prediction = model.predict(X)
        sample_submission = Directory.sample_submission
        sample_submission['deposit'] = prediction
        return sample_submission