import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor, GradientBoostingRegressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from utils.constant_utils import Config
from utils.common_utils import *
import inference
import model

def lightgbm(X, y, fitting : bool = True):
    lgb_model = lgb.LGBMRegressor(random_state=Config.RANDOM_SEED)
    if not fitting:
        return lgb_model
    
    lgb_model.fit(X, y)
    return lgb_model

def xgboost(X, y, fitting : bool = True):
    xgb_model = xgb.XGBRegressor(**Config.XGBOOST_PARAMS)
    if not fitting:
        return xgb_model
    xgb_model.fit(X, y)
    return xgb_model

def randomforest(X, y, fitting : bool = True):
    rf_model = RandomForestRegressor(**Config.RANDOM_FOREST_PARAMS)
    if not fitting:
        return rf_model
    rf_model.fit(X,y)
    return rf_model

def catboost(X, y, fitting : bool = True):
    cb_model = cb.CatBoostRegressor(**Config.CATBOOST_PARAMS)
    if not fitting:
        return cb_model
    cb_model.fit(X,y)
    return cb_model


## top modeling class

class TopNModeling():
    def __init__(self, train_data, valid_data, test_data, model_, origin_model):
        '''
        train_data = train_data
        valid_data = valid_data
        test_data = test_data
        model_ = trained model(한번 전체 데이터에 대해 학습시킨 모델)
        origin_model = 사용할 모델(ex. random forest, xgboost ...)
        
        '''
        self.train_data_ = train_data
        self.valid_data_ = valid_data
        self.test_data_ = test_data
        self.model_ = model_
        self.origin_model = origin_model
    
    def get_feature_importance(self):
        """
        학습된 모델의 feature importance를 계산하여 반환합니다.
        """
        importance = self.model_.feature_importances_
        
        train_df = self.train_data_.copy()
        train_df = self.train_df.drop(columns = ['deposit'], axis=1)
        
        importance_df = pd.DataFrame({
            'features': train_df.columns,
            'importance': importance
        }).sort_values(by='importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, importance_df):
        """
        feature importance를 시각화합니다.
        """
        plt.figure(figsize=(15, 12))
        plt.barh(importance_df['features'], importance_df['importance'], color='skyblue')
        plt.xlabel('importance')
        plt.ylabel('features')
        plt.title('feature importance plot')
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.show()
    
    def top_n_remodeling(self, n, importance_df):
        """
        가장 중요한 상위 n개의 feature를 사용하여 모델을 재학습하고, 
        MAE와 예측값을 반환합니다.
        """
        # Feature Importance에서 상위 n개의 feature 추출
        top_n_features = list(importance_df.head(n)['features'].values)

        # feature split
        train_data_n = self.train_data_[top_n_features + ['deposit']]
        valid_data_n = self.valid_data_[top_n_features + ['deposit']]
        test_data_n = self.test_data_[top_n_features]
        
        # 상위 n개의 feature만 사용하여 데이터를 필터링
        X_train, y_train, X_valid, y_valid, X_test = split_feature_target(train_data_n, valid_data_n, test_data_n)
        
        if self.origin_model == 'xgboost':
            new_model = model.xgboost(X_train, y_train)
        elif self.origin_model == 'lightgbm':
            new_model = model.lightgbm(X_train, y_train)
        elif self.origin_model == 'randomforest':
            new_model = model.randomforest(X_train, y_train)
        elif self.origin_model == 'catboost':
            new_model = model.catboost(X_train, y_train)
        
        prediction, mae = inference(new_model, 'validation', X_valid, y_valid)
    
        return prediction, mae