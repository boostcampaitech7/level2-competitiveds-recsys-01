from utils.constant_utils import Directory
from utils import common_utils
import preprocessing
import preprocessing_fn



import model
from inference import *
from utils.common_utils import submission_to_csv, mae_to_csv

def main():
    print("Start the main.py successfully!")
    df = common_utils.merge_data(Directory.train_data, Directory.test_data)
    train_data_, valid_data_, test_data_ = common_utils.train_valid_test_split(df)

    # 전처리 적용
    train_data_preprocessed = preprocessing.time_feature_preprocessing(train_data_)
    valid_data_preprocessed = preprocessing.time_feature_preprocessing(valid_data_)
    test_data_preprocessed = preprocessing.time_feature_preprocessing(test_data_)

    train_data_feat, valid_data_feat, test_data_feat = preprocessing_fn.create_clustering_target(train_data_preprocessed, valid_data_preprocessed, test_data_preprocessed)

    # 정규화
    train_data_scaled, valid_data_scaled, test_data_scaled = preprocessing_fn.standardization(train_data_feat, valid_data_feat, test_data_feat)

    # feature selection
    train_data_scaled, valid_data_scaled, test_data_scaled = preprocessing_fn.feature_selection(train_data_scaled, valid_data_scaled, test_data_scaled)

    # feature split
    X_train, y_train, X_valid, y_valid, X_test = common_utils.split_feature_target(train_data_scaled, valid_data_scaled, test_data_scaled)
    
    # train model
    print("Train the model")
    model_ = model.lightgbm(X_train, y_train)

    prediction, mae = inference(X_valid, y_valid, model_)

    test_pred = model_.predict(X_test)
    sample_submission = Directory.sample_submission
    sample_submission['deposit'] = test_pred
    submission_to_csv(sample_submission, 'feat_clustering')
    return prediction, mae




if __name__ == "__main__":
    prediction, mae = main()
    print(mae)