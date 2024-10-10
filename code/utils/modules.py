import pandas as pd
import time


def submission_to_csv(submit_df, file_name):
    file_name += '_' + time.strftime('%x', time.localtime())[:5].replace('/','') + '.csv'
    submit_df.to_csv('submit/'+file_name)

def mae_to_csv(mae_df, file_name):
    file_name += '_' + time.strftime('%x', time.localtime())[:5].replace('/','') + '.csv'
    mae_df.to_csv('mae_score/'+file_name)
