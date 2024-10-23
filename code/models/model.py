import lightgbm as lgb
import xgboost as xgb
from utils.constant_utils import Config

def lightgbm(X, y):
    lgb_model = lgb.LGBMRegressor(random_state=Config.RANDOM_SEED)
    lgb_model.fit(X, y)
    return lgb_model

def xgboost(X, y):
    xgb_model = xgb.XGBRegressor(learning_rate=0.3, n_estimators=1000, enable_categorical=True, random_state=Config.RANDOM_SEED)
    xgb_model.fit(X, y)
    return xgb_model