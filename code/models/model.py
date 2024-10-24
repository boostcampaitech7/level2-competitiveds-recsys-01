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
import numpy as np
import optuna
from utils import common_utils
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

def lightgbm(X, y, fitting : bool = True):
    lgb_model = lgb.LGBMRegressor(random_state=Config.RANDOM_SEED)
    if not fitting:
        return lgb_model
    
    lgb_model.fit(X, y)
    return lgb_model


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
    
def xgboost(X_train, y_train, X_valid, y_valid, optimize=False):
    
    if optimize:
        def objective(trial):
           
            params = {
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'lambda': trial.suggest_float('lambda', 0.1, 1.0),
                'alpha': trial.suggest_float('alpha', 0.1, 1.0)
            }

            model = xgb.XGBRegressor(**params, random_state=Config.RANDOM_SEED, enable_categorical=True)
            
            model.fit(X_train, y_train)
            
            valid_pred = model.predict(X_valid)

            return mean_absolute_error(y_valid, valid_pred)

        print("Optimizing XGBoost parameters...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50, n_jobs=-1, show_progress_bar=True)

        best_params = study.best_params
        print(f"Best parameters: {best_params}")
        
        xgb_model = xgb.XGBRegressor(**best_params, random_state=Config.RANDOM_SEED, enable_categorical=True)
    
    else:
        print("default parameters...")
        xgb_model = xgb.XGBRegressor(**Config.XGBOOST_PARAMS)
    
    xgb_model.fit(X_train, y_train)
    
    return xgb_model


# 하이퍼파라미터 최적화 함수
def objective(trial, model_name, X_selected, y):
    if model_name.startswith('lgb'):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 700),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': Config.RANDOM_SEED,
            'enable_categorical': True
        }
        if model_name == 'lgb_goss':
            params['boosting_type'] = 'goss'
    elif model_name == 'xgboost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 700),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }

    kf = KFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_SEED)
    fold_mae = []
    oof_predictions = np.zeros(len(X_selected))

    for train_idx, valid_idx in kf.split(X_selected):
        x_train, x_valid = X_selected.iloc[train_idx], X_selected.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        if model_name.startswith('lgb'):
            model = lgb.LGBMRegressor(**params)
        elif model_name == 'xgboost':
            model = xgb.XGBRegressor(**params, random_state=Config.RANDOM_SEED, enable_categorical=True)

        model.fit(x_train, y_train)
        oof_predictions[valid_idx] = model.predict(x_valid)

        fold_mae.append(mean_absolute_error(y_valid, oof_predictions[valid_idx]))

    return np.mean(fold_mae)

# 모델별 하이퍼파라미터 최적화 및 OOF 예측
def optimize_and_predict(X_selected, y, test_data_selected, models, saved_best_params):
    val_predictions = []
    test_predictions = []

    for model_name in models:
        print(f"Optimizing {model_name}...")

        if model_name in saved_best_params:
            best_params = saved_best_params[model_name]
        else:
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, model_name, X_selected, y), n_trials=30, n_jobs=-1, show_progress_bar=True)
            best_params = study.best_params
            print(f"Best parameters for {model_name}: {best_params}")

        oof_preds = np.zeros(len(X_selected))
        test_preds_fold = np.zeros(len(test_data_selected))

        kf = KFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_SEED)

        for train_idx, valid_idx in kf.split(X_selected):
            x_train, x_valid = X_selected.iloc[train_idx], X_selected.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            if model_name.startswith('lgb'):
                model = lgb.LGBMRegressor(**best_params)
            elif model_name == 'xgboost':
                model = xgb.XGBRegressor(**best_params, random_state=Config.RANDOM_SEED, enable_categorical=True)

            model.fit(x_train, y_train)
            oof_preds[valid_idx] = model.predict(x_valid)
            test_preds_fold += model.predict(test_data_selected) / kf.n_splits

        val_mae = mean_absolute_error(y, oof_preds)
        print(f"Validation MAE for {model_name}: {val_mae}")

        val_predictions.append(oof_preds)
        test_predictions.append(test_preds_fold)

        hyperparams = "stacking"
        common_utils.mae_to_csv('lim', 'stacking', hyperparams=hyperparams, mae=val_mae)

    return np.column_stack(val_predictions), np.column_stack(test_predictions)

# 스태킹 모델 학습 및 예측
def stack_models(val_predictions, y, test_predictions):
    stacking_model = LinearRegression()
    stacking_model.fit(val_predictions, y)

    return stacking_model.predict(test_predictions)
