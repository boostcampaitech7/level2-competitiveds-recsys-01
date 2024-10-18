import os
import pandas as pd

class Config:
    RANDOM_SEED = 42
    SPLIT_YEAR_START = 202307
    SPLIT_YEAR_END = 202312
    CATEGORICAL_COLUMNS = ['contract_type','cluster','subway_info','park_info', 'school_info', 'distance_category','age_category', 'floor_category', 'area_category']
    NUMERICAL_COLUMNS = ['area_m2','contract_year_month','contract_day','floor','built_year','latitude','longitude','age','deposit']

    # RandomForest Regressor Parameters
    RANDOM_FOREST_PARAMS = {
        'n_estimators': 1000,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'bootstrap': True,
        'random_state': RANDOM_SEED,
        'n_jobs': -1 
    }
    
    # LightGBM Regressor Parameters
    LIGHT_GBM_PARAMS = {
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'max_depth': -1,
        'num_leaves': 31, 
        'min_child_samples': 20, 
        'subsample': 0.8,
        'colsample_bytree': 0.8, 
        'random_state': RANDOM_SEED,
        'n_jobs': -1 
    }
    
    # XGBoost Regressor Parameters
    XGBOOST_PARAMS = {'learning_rate' : 0.3,
                      'n_estimators' : 1000,
                      'enable_categorical' : True,
                      'random_state' : RANDOM_SEED
                      }
    
    # CatBoost Regressor Paramters
    CATBOOST_PARAMS = {'iterations': 1000,
                       'learning_rate': 0.1,
                       'depth': 6,
                       'loss_function': 'MAE',
                       'random_seed': RANDOM_SEED,
                       'cat_features': CATEGORICAL_COLUMNS,
                       'verbose': 100              
}

    
class Directory:
    root_path = "/data/ephemeral/home/"
    train_data = pd.read_csv(os.path.join(root_path, 'data/train.csv'))
    test_data = pd.read_csv(os.path.join(root_path, 'data/test.csv'))
    sample_submission = pd.read_csv(os.path.join(root_path, 'data/sample_submission.csv'))

    interest_rate = pd.read_csv(os.path.join(root_path, "data/interestRate.csv"))
    park_info = pd.read_csv(os.path.join(root_path, "data/parkInfo.csv"))
    school_info = pd.read_csv(os.path.join(root_path, "data/schoolinfo.csv"))
    subway_info = pd.read_csv(os.path.join(root_path, "data/subwayInfo.csv"))

    result_path = os.path.join(root_path, 'result')
