from utils.constant_utils import Directory, Config
from utils import common_utils

import preprocessing
from features.clustering_features import *
from features.count_features import *
from features.distance_features import *
from features.other_features import *

from tabtransformer import *
from dataset import *
from trainer import TabTransformerTrainer

from tqdm import tqdm

import model
from inference import *
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

name = 'eun'
title = 'final_seed(10)_ensemble_weight_matrix(k=10)_and_xgboost_test_and_optuna'

print("total data load ...")
df = common_utils.merge_data(Directory.train_data, Directory.test_data)

### 클러스터 피처 apply
print("clustering apply ...")
for info_df_name in ['subway_info', 'school_info', 'park_info']:
    info_df = getattr(Directory, info_df_name)  
    df = clustering(df, info_df, feat_name=info_df_name, n_clusters=20)

### 이상치 처리
print("start to cleaning outliers...")
df = preprocessing.handle_age_outliers(df)

### 데이터 분할
print("train, valid, test split for preprocessing & feature engineering ...")
train_data_, valid_data_, test_data_ = common_utils.train_valid_test_split(df)


### 데이터 전처리
print("start to preprocessing...")
# type 카테고리화
train_data_ = preprocessing.numeric_to_categoric(train_data_, 'contract_type', {0:'new', 1:'renew', 2:'unknown'})
valid_data_ = preprocessing.numeric_to_categoric(valid_data_, 'contract_type', {0:'new', 1:'renew', 2:'unknown'})
test_data_ = preprocessing.numeric_to_categoric(test_data_, 'contract_type', {0:'new', 1:'renew', 2:'unknown'})

# 중복 제거
train_data_ = preprocessing.handle_duplicates(train_data_)
valid_data_ = preprocessing.handle_duplicates(valid_data_)

# 로그 변환
#df = preprocessing_fn.log_transform(df, 'deposit')


### 피처 엔지니어링
print("start to feature engineering...")
# clustering_feature
print("create clustering features")
train_data, valid_data, test_data = create_clustering_target(train_data_, valid_data_, test_data_)

# distance_features
print("create distance features")
train_data, valid_data, test_data = distance_gangnam(train_data, valid_data, test_data)
train_data, valid_data, test_data = create_nearest_subway_distance(train_data, valid_data, test_data)
#train_data, valid_data, test_data = create_nearest_park_distance_and_area(train_data, valid_data, test_data)
#train_data, valid_data, test_data = create_nearest_school_distance(train_data, valid_data, test_data)
train_data, valid_data, test_data = weighted_subway_distance(train_data, valid_data, test_data)
#train_data, valid_data, test_data = create_nearest_park_distance_and_area(train_data, valid_data, test_data)

# other_features
print("create other features")
#train_data, valid_data, test_data = create_temporal_feature(train_data, valid_data, test_data)
#train_data, valid_data, test_data = create_sin_cos_season(train_data, valid_data, test_data)
train_data, valid_data, test_data = create_floor_area_interaction(train_data, valid_data, test_data)
train_data, valid_data, test_data = create_sum_park_area_within_radius(train_data, valid_data, test_data)
#train_data, valid_data, test_data = shift_interest_rate_function(train_data, valid_data, test_data)
#train_data, valid_data, test_data = categorization(train_data, valid_data, test_data, category = 'age')
#train_data, valid_data, test_data = categorization(train_data, valid_data, test_data, category = 'floor')
#train_data, valid_data, test_data = categorization(train_data, valid_data, test_data, category = 'area_m2')


# count_features
print("create count features")
#train_data, valid_data, test_data = transaction_count_function(train_data, valid_data, test_data)
# 위의 함수를 바로 실행하기 위한 구조 : data/transaction_data에 train/valid/test_transaction_{month}.txt 구조의 파일이 있어야함
train_data, valid_data, test_data = create_subway_within_radius(train_data, valid_data, test_data)
train_data, valid_data, test_data = create_school_within_radius(train_data, valid_data, test_data)
train_data, valid_data, test_data = create_school_counts_within_radius_by_school_level(train_data, valid_data, test_data)
#train_data, valid_data, test_data = create_place_within_radius(train_data, valid_data, test_data)



### feature drop(제거하고 싶은 feature는 drop_columns로 제거됨. contract_day에 원하는 column을 추가)
selected_columns = [
'distance_km',
'floor_area_interaction',
'high_schools_within_radius',
'subways_within_radius',
'built_year',
'subway_info',
'longitude',
'nearest_subway_distance_x',
'area_m2',
'middle_schools_within_radius',
'schools_within_radius',
'nearest_subway_distance_y',
'cluster',
'contract_type',
'distance_to_centroid',
'distance_category',
'contract_year_month',
'latitude',
'nearest_park_area_sum',
'elementary_schools_within_radius',
'deposit'
]

train_data_ = train_data_[selected_columns + ['deposit']]
valid_data_ = valid_data_[selected_columns + ['deposit']]
test_data_ = test_data_[selected_columns]

# 수치형, 범주형 분리
numeric_train_data, numeric_valid_data, numeric_test_data = split_to_numeric(train_data_, valid_data_, test_data_)
categoric_train_data, categoric_valid_data, categoric_test_data = split_to_categoric(train_data_, valid_data_, test_data_)

# 수치형 정규화
numeric_train_data, numeric_valid_data, numeric_test_data = preprocessing.standardization(numeric_train_data, numeric_valid_data, numeric_test_data, scaling_type = 'minmax')

# 범주형 임베딩
categoric_train_data = applying_embedding(categoric_train_data)
categoric_valid_data = applying_embedding(categoric_valid_data)
categoric_test_data = applying_embedding(categoric_test_data)

# 범주형 min-max
categoric_train_data, categoric_valid_data, categoric_test_data = categorical_scaler(categoric_train_data, categoric_valid_data, categoric_test_data)

X_train, y_train, X_valid, y_valid, X_test = concat_numeric_categoric(categoric_train_data, categoric_valid_data, categoric_test_data, numeric_train_data, numeric_valid_data, numeric_test_data)

train_loader, valid_loader, test_loader = get_dataloader(categoric_train_data, categoric_valid_data, categoric_test_data, 
                                                    numeric_train_data, numeric_valid_data, numeric_test_data)

# 모델 초기화
num_features = numeric_train_data.shape[1] - 1
cat_emb_dim = categoric_test_data.shape[1] - 1
model = TabTransformer(num_features=num_features, cat_emb_dim=cat_emb_dim, n_heads=Config.TRANSFORMER_CONFIG['n_heads'], n_layers=Config.TRANSFORMER_CONFIG['n_layers'])

# 4. 손실 함수와 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Config.TRANSFORMER_CONFIG['learning_rate'], weight_decay=Config.TRANSFORMER_CONFIG['weight_decay'])  # L2 정규화

model_ = TabTransformer(num_features=num_features, cat_emb_dim=16)

# Trainer 초기화
trainer = TabTransformerTrainer(model_, optimizer, criterion, Config.TRANSFORMER_CONFIG['device'])

# 학습 실행
X_train, y_train = next(iter(train_loader))
X_valid, y_valid = next(iter(valid_loader))

trainer.train(X_train, y_train, X_valid, y_valid, Config.TRANSFORMER_CONFIG['batch_size'], Config.TRANSFORMER_CONFIG['num_epochs'])

predictions, targets = trainer.inference(test_loader)

sample_submission = Directory.sample_submission
sample_submission['deposit'] = predictions



### save submission
common_utils.submission_to_csv(sample_submission, title)

print("Successfully executed main.py.")