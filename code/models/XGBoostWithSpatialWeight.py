import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

from utils.constant_utils import Config

class XGBoostWithSpatialWeight:
    def __init__(self, spatial_weight_matrix):
        self.spatial_weight_matrix = spatial_weight_matrix
        self.basic_model = xgb.XGBRegressor(learning_rate=0.3, n_estimators=1000, enable_categorical=True, random_state=Config.RANDOM_SEED)
        self.spatial_model = xgb.XGBRegressor(learning_rate=0.3, n_estimators=1000, enable_categorical=True, random_state=Config.RANDOM_SEED)

    def add_spatial_features(self, data_chunk, chunk_id, dataset_type):
        weight_matrix = self.spatial_weight_matrix.load_weight_matrix(chunk_id=chunk_id, dataset_type=dataset_type)
        if weight_matrix is None:
            return data_chunk

        # 훈련 데이터인 경우 deposit을 사용하여 공간적 피처를 생성
        if dataset_type in ['train', 'train_total']:
            spatial_features = weight_matrix.dot(data_chunk['deposit'].values)
        else:
            # 검증 및 테스트 데이터인 경우, 초기 예측값을 이용하여 공간적 피처를 생성
            if 'initial_preds' in data_chunk.columns:
                spatial_features = weight_matrix.dot(data_chunk['initial_preds'].values)
            else:
                # 초기 예측값이 없으면 공간적 피처를 생성할 수 없음
                spatial_features = np.zeros(len(data_chunk))  # 적절한 대체값 사용

        data_chunk['spatial_feature'] = spatial_features

        return data_chunk

    def train(self, train_data, dataset_type):
        # 모델 훈련을 위한 피처와 타겟 분리
        x_train = train_data.drop(columns=['deposit'])
        y_train = train_data['deposit']
        
        # 초기 모델 훈련
        self.basic_model.fit(x_train, y_train)

        num_chunks = len(train_data) // self.spatial_weight_matrix.chunk_size + 1
        train_data_with_spatial = pd.DataFrame()  # 초기화

        for i in range(num_chunks):
            chunk = train_data[i * self.spatial_weight_matrix.chunk_size: (i + 1) * self.spatial_weight_matrix.chunk_size]
            if chunk.empty:
                continue
            
            # 공간적 피처 추가
            chunk_with_spatial = self.add_spatial_features(chunk.copy(), chunk_id=i, dataset_type=dataset_type)
            train_data_with_spatial = pd.concat([train_data_with_spatial, chunk_with_spatial], ignore_index=True)

        # 최종 훈련 데이터에서 모델 훈련
        self.spatial_model.fit(train_data_with_spatial.drop(columns=['deposit']), train_data_with_spatial['deposit'])

    def evaluate(self, valid_data):
        # 초기 예측값 생성
        initial_preds = self.basic_model.predict(valid_data.drop(columns=['deposit']))
        valid_data['initial_preds'] = initial_preds

        # 공간적 피처 추가
        num_valid_chunks = len(valid_data) // self.spatial_weight_matrix.chunk_size + 1
        valid_data_with_spatial = pd.DataFrame()

        for j in range(num_valid_chunks):
            valid_chunk = valid_data[j * self.spatial_weight_matrix.chunk_size: (j + 1) * self.spatial_weight_matrix.chunk_size]
            if valid_chunk.empty:
                continue
            
            valid_chunk_with_spatial = self.add_spatial_features(valid_chunk.copy(), chunk_id=j, dataset_type='valid')
            valid_data_with_spatial = pd.concat([valid_data_with_spatial, valid_chunk_with_spatial], ignore_index=True)

        # 최종 검증 데이터에 대한 예측
        final_preds = self.spatial_model.predict(valid_data_with_spatial.drop(columns=['deposit', 'initial_preds']))

        # MAE 계산
        mae = mean_absolute_error(valid_data_with_spatial['deposit'], final_preds)
        print(f"MAE on validation data: {mae}")
        return mae

    def inference(self, test_data):
        # 초기 예측값 생성
        initial_preds = self.basic_model.predict(test_data.drop(columns=['deposit'], errors='ignore'))
        test_data['initial_preds'] = initial_preds

        # 공간적 피처 추가
        num_test_chunks = len(test_data) // self.spatial_weight_matrix.chunk_size + 1
        test_data_with_spatial = pd.DataFrame()

        for j in range(num_test_chunks):
            test_chunk = test_data[j * self.spatial_weight_matrix.chunk_size: (j + 1) * self.spatial_weight_matrix.chunk_size]
            if test_chunk.empty:
                continue
            
            test_chunk_with_spatial = self.add_spatial_features(test_chunk.copy(), chunk_id=j, dataset_type='test')
            test_data_with_spatial = pd.concat([test_data_with_spatial, test_chunk_with_spatial], ignore_index=True)

        # 최종 테스트 데이터에 대한 예측
        final_preds = self.spatial_model.predict(test_data_with_spatial.drop(columns=['deposit', 'initial_preds']))

        return final_preds