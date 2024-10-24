from torch.utils.data import Dataset
from utils.constant_utils import Directory
from utils import common_utils
from handler import preprocessing
from features.clustering_features import *
from features.count_features import *
from features.distance_features import *
from features.other_features import *

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn
import torch.optim as optim

from models.CombinedModel import *
from models.inference import *

def create_embedding(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_df = pd.concat([train_data, valid_data, test_data], axis=0)
    total_df = total_df[~total_df.index.duplicated(keep='first')]
    categorical_cols = [col for col in total_df.columns if total_df[col].dtype not in ['int32', 'int64', 'float32', 'float64']]
    
    embedding_modules = {
            col: nn.Embedding(
                num_embeddings= total_df[col].nunique(),
                embedding_dim = min(50, int((total_df[col].nunique()+1)/2))

                )
        for col in categorical_cols
    }

    train_data.reset_index(inplace=True)
    valid_data.reset_index(inplace=True)
    test_data.reset_index(inplace=True)

    for col in categorical_cols:
        # 카테고리를 인덱스로 매핑한다.
        indexer = {v: i for i, v in enumerate(total_df[col].unique())}
        train_data[f"{col}_embedded"] = total_df[col].map(indexer)
        valid_data[f"{col}_embedded"] = total_df[col].map(indexer)
        test_data[f"{col}_embedded"] = total_df[col].map(indexer)

        # LongTensor로 변환
        train_embedded = embedding_modules[col](torch.LongTensor(train_data[f"{col}_embedded"].values))
        valid_embedded = embedding_modules[col](torch.LongTensor(valid_data[f"{col}_embedded"].values))
        test_embedded = embedding_modules[col](torch.LongTensor(test_data[f"{col}_embedded"].values))

        # 기존 범주형 변수 drop
        train_data.drop(columns=[col], inplace=True)
        valid_data.drop(columns=[col], inplace=True)        
        test_data.drop(columns=[col], inplace=True)
        
        # 텐서를 detach 후 numpy로 변환하여 새로운 컬럼으로 적용한다.
        train_data[f"{col}_embedded"] = train_embedded.detach().numpy()
        valid_data[f"{col}_embedded"] = valid_embedded.detach().numpy()
        test_data[f"{col}_embedded"] = test_embedded.detach().numpy()
    
    return (train_data, valid_data, test_data)

#위, 경도 데이터로 피처맵을 생성하는 데이터셋입니다.
class GridDataset(Dataset):
    def __init__(self, mode="train"):
        self.mode = mode
        total_df = common_utils.merge_data(Directory.train_data, Directory.test_data)
        total_df = preprocessing.handle_age_outliers(total_df)
        
        lat_bins = np.linspace(total_df['latitude'].min(), total_df['latitude'].max(), 43)
        long_bins = np.linspace(total_df['longitude'].min(), total_df['longitude'].max(), 28)

        total_df['lat_bin'] = np.digitize(total_df['latitude'], lat_bins)
        total_df['long_bin'] = np.digitize(total_df['longitude'], long_bins)

        train_data_, valid_data_, test_data_ = common_utils.train_valid_test_split(total_df)

        # 중복 행 제거
        train_data_ = preprocessing.handle_duplicates(train_data_)
        valid_data_ = preprocessing.handle_duplicates(valid_data_)
        

        train_data_.reset_index(drop=True, inplace=True)
        valid_data_.reset_index(drop=True, inplace=True)
        test_data_.reset_index(drop=True, inplace=True)

        grid = torch.zeros(len(lat_bins), len(long_bins))

        if self.mode=="train":
            # train + valid
            X = torch.stack([grid]*len(train_data_))
            y = torch.tensor(train_data_['deposit'].values)
    
            for i, data in tqdm(train_data_.iterrows()):
                # 아파트 위치의 bin이 1부터 시작해서 -1 해준다.
                lat_bin, long_bin = data['lat_bin']-1, data['long_bin']-1
                X[i,lat_bin,long_bin] = 1

            self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        elif mode=='valid':
            X = torch.stack([grid]*len(valid_data_))
            y = torch.tensor(valid_data_['deposit'].values)
            
            for i, data in tqdm(valid_data_.iterrows()):
                # 아파트 위치의 bin이 1부터 시작해서 -1 해준다.
                lat_bin, long_bin = data['lat_bin']-1, data['long_bin']-1
                X[i,lat_bin,long_bin] = 1

            self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        else:
            X = torch.stack([grid]*len(test_data_))
            for i, data in tqdm(test_data_.iterrows()):
                # 아파트 위치의 bin이 1부터 시작해서 -1 해준다.
                lat_bin, long_bin = data['lat_bin']-1, data['long_bin']-1
                X[i,lat_bin,long_bin] = 1

            self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.mode=="train" or self.mode=="valid":
            return (self.X[idx], self.y[idx])
        else:
            return (self.X[idx])

# 위, 경도 이외의 다른 데이터들을 불러옵니다.
class MLPDataset(Dataset):
    def __init__(self, mode="train"):
        self.mode = mode
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

        ### 피처 엔지니어링
        print("start to feature engineering...")
        # clustering_feature
        print("create clustering features")
        train_data, valid_data, test_data = create_clustering_target(train_data_, valid_data_, test_data_)

        # distance_features
        print("create distance features")
        train_data, valid_data, test_data = distance_gangnam(train_data, valid_data, test_data)
        train_data, valid_data, test_data = create_nearest_subway_distance(train_data, valid_data, test_data)
        train_data, valid_data, test_data = create_nearest_park_distance_and_area(train_data, valid_data, test_data)
        train_data, valid_data, test_data = create_nearest_school_distance(train_data, valid_data, test_data)
        train_data, valid_data, test_data = weighted_subway_distance(train_data, valid_data, test_data)
        train_data, valid_data, test_data = create_nearest_park_distance_and_area(train_data, valid_data, test_data)

        # other_features
        print("create other features")
        train_data, valid_data, test_data = create_temporal_feature(train_data, valid_data, test_data)
        train_data, valid_data, test_data = create_sin_cos_season(train_data, valid_data, test_data)
        train_data, valid_data, test_data = create_floor_area_interaction(train_data, valid_data, test_data)
        train_data, valid_data, test_data = create_sum_park_area_within_radius(train_data, valid_data, test_data)
        train_data, valid_data, test_data = shift_interest_rate_function(train_data, valid_data, test_data)
        train_data, valid_data, test_data = categorization(train_data, valid_data, test_data, category = 'age')
        train_data, valid_data, test_data = categorization(train_data, valid_data, test_data, category = 'floor')
        train_data, valid_data, test_data = categorization(train_data, valid_data, test_data, category = 'area_m2')


        # count_features
        print("create count features")
        train_data, valid_data, test_data = create_subway_within_radius(train_data, valid_data, test_data)
        train_data, valid_data, test_data = create_school_within_radius(train_data, valid_data, test_data)
        train_data, valid_data, test_data = create_school_counts_within_radius_by_school_level(train_data, valid_data, test_data)
        train_data, valid_data, test_data = create_place_within_radius(train_data, valid_data, test_data)

        # top 20개의 column만 Input으로 제한 - 사용 시 주석을 해제
        top20_cols = ['distance_km', 'built_year', 'area_m2', 
                'cluster', 'contract_year_month', 'floor_area_interaction', 
                'subway_info', 'middle_schools_within_radius', 'contract_type', 
                'high_schools_within_radius', 'elementary_schools_within_radius', 'nearest_subway_distance_x', 'nearest_subway_distance_y',
                'distance_category', 'subways_within_radius', 'nearest_park_area_sum',
                'schools_within_radius','distance_to_centroid', 'deposit']

        train_data_, valid_data_, test_data_ = train_data[top20_cols], valid_data[top20_cols], test_data[top20_cols]

        # categorical embedding 생성
        train_data_, valid_data_, test_data_ = create_embedding(train_data_, valid_data_, test_data_)

        ### 정규화
        print("standardization...")
        train_data_, valid_data_, test_data_ = preprocessing.standardization(train_data_, valid_data_, test_data_, scaling_type = 'standard')

        # feature split
        X_train, y_train, X_valid, y_valid, X_test = common_utils.split_feature_target(train_data_, valid_data_, test_data_)

        if mode=='train':
            # train + valid
            self.X = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)
            self.y = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

        elif mode=='valid':
            self.X = torch.tensor(X_valid.values, dtype=torch.float32).unsqueeze(1)
            self.y = torch.tensor(y_valid.values, dtype=torch.float32).unsqueeze(1)

        else:
            self.X = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.mode=="train" or self.mode=="valid":
            return (self.X[idx], self.y[idx])
        else:
            return (self.X[idx])

class CombinedDataset(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        self.cnn = GridDataset(mode)
        self.mlp = MLPDataset(mode)

    def __len__(self):
        return len(self.cnn)
    
    def __getitem__(self, idx):
        if self.mode == 'train' or self.mode=='valid':
            X_cnn, y = self.cnn[idx]
            X_mlp, y = self.mlp[idx]
            return (X_cnn,X_mlp, y)
        else:
            X_cnn = self.cnn[idx]
            X_mlp = self.mlp[idx]
            return (X_cnn, X_mlp)
