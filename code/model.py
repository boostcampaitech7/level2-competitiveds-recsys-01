import lightgbm as lgb
from utils.constant_utils import Config

def lightgbm(X, y):
    lgb_model = lgb.LGBMRegressor(random_state=Config.RANDOM_SEED)
    lgb_model.fit(X, y)
    return lgb_model

# def xgboost(X, y):
#     lgb_model = lgb.LGBMRegressor(random_state=Config.RANDOM_SEED)
#     lgb_model.fit(X, y)
#     return lgb_model

# def catboost(X, y):
#     lgb_model = lgb.LGBMRegressor(random_state=Config.RANDOM_SEED)
#     lgb_model.fit(X, y)
#     return lgb_model