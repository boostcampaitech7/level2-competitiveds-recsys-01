import numpy as np
from utils.constant_utils import Config

class SeedEnsemble:
    def __init__(self, model_class, spatial_weight_matrix, seeds=Config.SEEDS):
        self.model_class = model_class
        self.spatial_weight_matrix = spatial_weight_matrix
        self.seeds = seeds
        self.models = []

    def train(self, train_data, dataset_type):
        for seed in self.seeds:
            model_ = self.model_class(self.spatial_weight_matrix, seed=seed)
            print(f"train model seed = {seed}")
            model_.train(train_data, dataset_type=dataset_type)
            self.models.append(model_)

    def evaluate(self, valid_data, train_data):
        xgb_test_preds = []
        for model in self.models:
            preds, _ = model.evaluate(valid_data, train_data)
            xgb_test_preds.append(preds)

        # 예측 값 평균으로 최종 예측값 계산
        final_preds = np.round(np.mean(xgb_test_preds, axis=0))

        return final_preds

    def inference(self, test_data, train_data):
        xgb_test_preds = []
        for model in self.models:
            preds = model.inference(test_data, train_data)
            xgb_test_preds.append(preds)

        # 예측 값 평균으로 최종 예측값 계산
        final_preds = np.round(np.mean(xgb_test_preds, axis=0))

        return final_preds