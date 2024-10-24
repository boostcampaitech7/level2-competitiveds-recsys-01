from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torch
import numpy as np

from utils.constant_utils import Directory, Config

def split_to_numeric(train_data, valid_data, test_data):
    def calculator(df):
        df = df.select_dtypes(exclude = ['category'])
        return df
    
    train_data = calculator(train_data)
    valid_data = calculator(valid_data)
    test_data = calculator(test_data)

    return train_data, valid_data, test_data

def split_to_categoric(train_data, valid_data, test_data):
    def calculator(df):
        df = df.select_dtypes(include = ['category'])
        return df
    
    train_data = calculator(train_data)
    valid_data = calculator(valid_data)
    test_data = calculator(test_data)

    return train_data, valid_data, test_data
    

def applying_embedding(df):
    for column in df.columns:
        indexer = {v : i for i, v in enumerate(df[column].unique())}
        df[column] = df[column].map(indexer)

    def category_embedding_modules(df):
        category_embedding_module_dict = {}
        for column in df.columns:
            num_category = len(set(df[column]))
            embedding_dim = min(50,max(1,round(num_category/2)))
            module = nn.Embedding(num_embeddings=num_category, embedding_dim=embedding_dim)
            category_embedding_module_dict[column] = module

        return category_embedding_module_dict
    
    category_embedding_module_dict = category_embedding_modules(df)
    embeded_vector_list = []
    for column in df.columns:
        module = category_embedding_module_dict[column]
        embeded_vector = module(torch.tensor(df[column].tolist()))
        embeded_vector_list.append(embeded_vector.detach().numpy())

    embedded_vector = np.hstack(embeded_vector_list)
    return embedded_vector


def categorical_scaler(categoric_train_data, categoric_valid_data, categoric_test_data):
    scaler = MinMaxScaler()
    categoric_train_data = scaler.fit_transform(categoric_train_data)
    categoric_valid_data = scaler.transform(categoric_valid_data)
    categoric_test_data = scaler.transform(categoric_test_data)

    return categoric_train_data, categoric_valid_data, categoric_test_data

def concat_numeric_categoric(categoric_train_data, categoric_valid_data, categoric_test_data, numeric_train_data, numeric_valid_data, numeric_test_data):
    X_train = np.hstack([numeric_train_data.drop(['deposit'], axis = 1).values, categoric_train_data])
    y_train = numeric_train_data['deposit']

    X_valid = np.hstack([numeric_valid_data.drop(['deposit'], axis = 1).values, categoric_valid_data])
    y_valid = numeric_valid_data['deposit']

    X_test = np.hstack([numeric_test_data.values, categoric_test_data])

    return X_train, y_train, X_valid, y_valid, X_test


def get_dataloader(categoric_train_data, categoric_valid_data, categoric_test_data, 
                  numeric_train_data, numeric_valid_data, numeric_test_data):
    # 1. 데이터 준비: 예시 데이터 (수치형과 범주형 분리)
    X_num_train = torch.tensor(numeric_train_data.drop(['deposit'], axis=1).values, dtype=torch.float32)  # 수치형
    X_cat_train = torch.tensor(categoric_train_data, dtype=torch.float32)  # 임베딩된 범주형
    y_train = torch.tensor(numeric_train_data['deposit'].values, dtype=torch.float32).unsqueeze(1)
    
    X_num_valid = torch.tensor(numeric_valid_data.drop(['deposit'], axis=1).values, dtype=torch.float32)
    X_cat_valid = torch.tensor(categoric_valid_data, dtype=torch.float32)
    y_valid = torch.tensor(numeric_valid_data['deposit'].values, dtype=torch.float32).unsqueeze(1)
    
    X_num_test = torch.tensor(numeric_test_data.values, dtype=torch.float32)
    X_cat_test = torch.tensor(categoric_test_data, dtype=torch.float32)
    
    # 2. DataLoader 생성
    train_dataset = TensorDataset(X_num_train, X_cat_train, y_train)
    valid_dataset = TensorDataset(X_num_valid, X_cat_valid, y_valid)
    test_dataset = TensorDataset(X_num_test, X_cat_test)  # y값 없이 생성
    
    train_loader = DataLoader(train_dataset, batch_size=Config.TRANSFORMER_CONFIG['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.TRANSFORMER_CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=Config.TRANSFORMER_CONFIG['batch_size'])
    
    return train_loader, valid_loader, test_loader