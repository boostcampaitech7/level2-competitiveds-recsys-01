import lightgbm as lgb
import xgboost as xgb
from utils.constant_utils import Config
import optuna
from sklearn.metrics import mean_absolute_error
from optuna.samplers import TPESampler

def lightgbm(X, y):
    lgb_model = lgb.LGBMRegressor(random_state=Config.RANDOM_SEED)
    lgb_model.fit(X, y)
    return lgb_model

def xgboost(X, y):
    #xgb_model = xgb.XGBRegressor(learning_rate=0.3, n_estimators=1000, enable_categorical=True, random_state=Config.RANDOM_SEED)
    xgb_model = xgb.XGBRegressor(learning_rate=0.12243929663868629, n_estimators= 648, max_depth= 10, min_child_weight= 2, gamma= 0.11673592777053933, subsample= 0.999075058215622, colsample_bytree= 0.8848954245105137,enable_categorical=True, random_state=Config.RANDOM_SEED)
    xgb_model.fit(X, y)
    return xgb_model

# def catboost(X, y):
#     lgb_model = lgb.LGBMRegressor(random_state=Config.RANDOM_SEED)
#     lgb_model.fit(X, y)
#     return lgb_model

def objective(trial, X, y):
    # 하이퍼파라미터를 설정
    param = {
        'verbosity': 0,
        'objective': 'reg:squarederror',
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': Config.RANDOM_SEED,
        'enable_categorical': True
    }

    # XGBoost 모델 생성 및 학습
    model = xgb.XGBRegressor(**param)
    model.fit(X, y)
    
    # 검증 세트로 MAE 계산 (교차 검증 또는 홀드아웃 방법도 가능)
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)

    return mae

def xgboost_optuna(X, y):
    # Optuna 스터디 생성 (TPE Sampler 사용)
    sampler = TPESampler(seed=Config.RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # 최적의 하이퍼파라미터 탐색
    study.optimize(lambda trial: objective(trial, X, y), n_trials=30)

    print("Best trial: ", study.best_trial.params)
    
    # 최적의 하이퍼파라미터로 최종 모델 학습
    best_params = study.best_trial.params
    model = xgb.XGBRegressor(**best_params)
    model.fit(X, y)

    return model
