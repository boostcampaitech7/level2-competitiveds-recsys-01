import os
import pandas as pd

class Config:
    RANDOM_SEED = 42

class Directory:
    root_path = "/data/ephemeral/home/level2-competitiveds-recsys-01"
    train_data = pd.read_csv(os.path.join(root_path, 'data/train.csv'))
    test_data = pd.read_csv(os.path.join(root_path, 'data/test.csv'))
    sample_submission = pd.read_csv(os.path.join(root_path, 'data/sample_submission.csv'))

    interest_rate = pd.read_csv(os.path.join(root_path, "data/interestRate.csv"))
    park_info = pd.read_csv(os.path.join(root_path, "data/parkInfo.csv"))
    school_info = pd.read_csv(os.path.join(root_path, "data/schoolinfo.csv"))
    subway_info = pd.read_csv(os.path.join(root_path, "data/subwayInfo.csv"))

    result_path = os.path.join(root_path, 'result')