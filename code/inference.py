import os
import time
import numpy as np

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


def log_inference(model, type, X, y=None):
    
    if type=='validation':
        prediction = model.predict(X).astype(np.float64)
        prediction = np.exp(prediction)
        
        if np.any(np.isnan(prediction)):
            raise ValueError("Prediction contains NaN values. Check your model or input data.")

        if np.any(np.isinf(prediction)):
            raise ValueError("Prediction contains Inf values. This may be due to numerical instability in the model.")

        if np.any(prediction <= 0):
            zero_indices = np.where(prediction <= 0)[0]
            raise ValueError(f"Prediction contains zero or negative values at indices: {zero_indices}. Clipping these values to avoid issues.")
        
        mae = mean_absolute_error(y, prediction)
        return prediction, mae

    prediction = model.predict(X).astype(np.float64)
    prediction = np.exp(prediction)
    sample_submission = Directory.sample_submission
    sample_submission['deposit'] = prediction

    return sample_submission
