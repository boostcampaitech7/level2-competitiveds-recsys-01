import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import joblib
import os
from sklearn.neighbors import BallTree
import math

from utils.constant_utils import Directory

class SpatialWeightMatrix:
    def __init__(self, k=10, chunk_size=10000):
        self.k = k
        self.chunk_size = chunk_size
        self.n = 0
        self.base_save_directory = Directory.root_path + 'data/spatial_matrix'
        os.makedirs(self.base_save_directory, exist_ok=True)

    def get_save_directory(self, dataset_type):
        '''
        dataset_type에 따라 지정된 디렉터리 경로 반환
        dataset_type : train, valid, test, train_total # train_total : train + valid 통합해서 훈련 시 사용
        '''
        dir_path = os.path.join(self.base_save_directory, dataset_type)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def create_weight_matrix(self, data_chunk, chunk_id, dataset_type, tree):
        '''
        청크 별로 공간적 가중치 행렬 생성
        '''
        file_path = os.path.join(self.get_save_directory(dataset_type), f'weight_matrix_chunk_{chunk_id}.pkl')

        # 이미 파일이 존재하면 생성하지 않고 넘어감
        if os.path.exists(file_path):
            return
        
        coords = data_chunk[['latitude', 'longitude']].values
        distance, indices = tree.query(coords, k=self.k)

        weight_matrix = np.zeros((len(data_chunk), self.n))
        for i in range(len(data_chunk)):
            weights = np.zeros(self.k)
            for j in range(self.k):
                dist = distance[i, j]
                weights[j] = 1 / (dist + 1e-10) # 거리에 따른 가중치
            weights /= weights.sum()

            for j in range(self.k):
                weight_matrix[i, indices[i, j]] = weights[j]

        sparse_matrix = csr_matrix(weight_matrix) # 생성된 공간적 가중치 행렬을 희소 행렬로 저장
        joblib.dump(sparse_matrix, file_path)

    def generate_weight_matrices(self, full_data, train_data, dataset_type):

        tree = BallTree(train_data[['latitude', 'longitude']], metric='haversine') # train data에 대한 ball tree 생성
        self.n = len(train_data)

        # 전체 데이터를 청크로 분할
        num_chunks = math.ceil(len(full_data) / self.chunk_size)
        for i in range(num_chunks):
            chunk = full_data[i * self.chunk_size: (i + 1) * self.chunk_size]
            
            if chunk.empty:
                continue
            
            self.create_weight_matrix(chunk, i, dataset_type, tree) # 청크 별로 공간적 가중치 행렬 생성

    def load_weight_matrix(self, chunk_id, dataset_type):
        '''
        공간적 가중치 행렬 불러오기
        '''
        try:
            return joblib.load(os.path.join(self.get_save_directory(dataset_type), f'weight_matrix_chunk_{chunk_id}.pkl'))
        except FileNotFoundError:
            print(f"Weight matrix for chunk {chunk_id} in {dataset_type} not found.")
            return None