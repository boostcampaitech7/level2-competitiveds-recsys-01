import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import math
import json

from utils.constant_utils import Directory

class XGBoostWithSpatialWeight:

    def __init__(self, spatial_weight_matrix, seed):
        path = Directory.root_path + 'code/models/config_optuna.json'
        with open(path, 'r') as file:
            config = json.load(file)
        hyperparams = config['model']['hyperparameters']  
        self.seed = seed
        self.spatial_weight_matrix = spatial_weight_matrix
        self.spatial_model = xgb.XGBRegressor(**hyperparams, random_state=self.seed)

    def add_spatial_features(self, data_chunk, area_per_deposit, chunk_id, dataset_type):
        weight_matrix = self.spatial_weight_matrix.load_weight_matrix(chunk_id=chunk_id, dataset_type=dataset_type)
        if weight_matrix is None:
            return data_chunk
        
        spatial_features = weight_matrix.dot(area_per_deposit)

        data_chunk['spatial_feature'] = spatial_features

        return data_chunk

    def train(self, train_data, dataset_type):
        x_train = train_data.drop(columns=['deposit'])
        y_train = train_data['deposit']
        area_per_deposit = y_train / x_train['area_m2'] # 면적 당 전세가

        num_chunks = math.ceil(len(train_data) / self.spatial_weight_matrix.chunk_size)
        train_data_with_spatial = pd.DataFrame()

        chunks = []
        for i in range(num_chunks):
            chunk = train_data[i * self.spatial_weight_matrix.chunk_size: (i + 1) * self.spatial_weight_matrix.chunk_size]
            if chunk.empty:
                continue
            
            # 전세가의 가중 평균 피처 추가
            chunk_with_spatial = self.add_spatial_features(chunk.copy(), area_per_deposit, chunk_id=i, dataset_type=dataset_type)
            chunks.append(chunk_with_spatial)
        train_data_with_spatial = pd.concat(chunks, ignore_index=True)

        self.spatial_model.fit(train_data_with_spatial.drop(columns=['deposit']), train_data_with_spatial['deposit'])

    def evaluate(self, valid_data, train_data):
        area_per_deposit = train_data['deposit'] / train_data['area_m2'] # 면적 당 전세가

        num_valid_chunks = math.ceil(len(valid_data) / self.spatial_weight_matrix.chunk_size)
        valid_data_with_spatial = pd.DataFrame()

        chunks = []
        for j in range(num_valid_chunks):
            valid_chunk = valid_data[j * self.spatial_weight_matrix.chunk_size: (j + 1) * self.spatial_weight_matrix.chunk_size]
            if valid_chunk.empty:
                continue
            
            # 전세가의 가중 평균 피처 추가
            valid_chunk_with_spatial = self.add_spatial_features(valid_chunk.copy(), area_per_deposit, chunk_id=j, dataset_type='valid')
            chunks.append(valid_chunk_with_spatial)
        valid_data_with_spatial = pd.concat(chunks, ignore_index=True)

        # 최종 검증 데이터에 대한 예측
        final_preds = self.spatial_model.predict(valid_data_with_spatial.drop(columns=['deposit']))

        mae = mean_absolute_error(valid_data_with_spatial['deposit'], final_preds)
        print(f"MAE on validation data: {mae}")
        
        return final_preds, mae

    def inference(self, test_data, train_data):

        area_per_deposit = train_data['deposit'] / train_data['area_m2']

        num_test_chunks = math.ceil(len(test_data) / self.spatial_weight_matrix.chunk_size)
        test_data_with_spatial = pd.DataFrame()

        chunks = []
        for j in range(num_test_chunks):
            test_chunk = test_data[j * self.spatial_weight_matrix.chunk_size: (j + 1) * self.spatial_weight_matrix.chunk_size]
            if test_chunk.empty:
                continue
            
            # 전세가의 가중 평균 피처 추가
            test_chunk_with_spatial = self.add_spatial_features(test_chunk.copy(), area_per_deposit, chunk_id=j, dataset_type='test')
            chunks.append(test_chunk_with_spatial)
        test_data_with_spatial = pd.concat(chunks, ignore_index=True)
        print(f"inference model seed = {self.seed}")

        # 최종 테스트 데이터에 대한 예측
        final_preds = self.spatial_model.predict(test_data_with_spatial.drop(columns=['deposit']))

        return final_preds