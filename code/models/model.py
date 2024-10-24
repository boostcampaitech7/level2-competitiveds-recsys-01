import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from utils.constant_utils import Config
import optuna
from sklearn.metrics import mean_absolute_error
from utils import common_utils
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

def lightgbm(X, y, fitting : bool = True):
    lgb_model = lgb.LGBMRegressor(random_state=Config.RANDOM_SEED)
    if not fitting:
        return lgb_model
    
    lgb_model.fit(X, y)
    return lgb_model

def catboost(X, y, fitting : bool = True):
    cb_model = cb.CatBoostRegressor(**Config.CATBOOST_PARAMS)
    if not fitting:
        return cb_model
    cb_model.fit(X,y)
    return cb_model

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
