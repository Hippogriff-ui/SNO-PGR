import numpy as np
import pandas as pd
import sys

def get_remained_feature_index(pcc_mat,th):
    n = len(pcc_mat)
    remain = np.ones(n)
    res_ind= []
    for i in range(n):
        if remain[i] == 0:
            continue
        for j in range(i+1,n):
            if remain[j] ==0:
                continue
            if np.isnan(pcc_mat[i][j]):
                remain[j]=0
            if np.abs(pcc_mat[i][j])>th:
                remain[j]=0
    for i in range(n):
        if remain[i]==1:
            res_ind.append(i)
    return res_ind

def pcc_filter(input_path, final_score_file, th):
    data = pd.read_csv(input_path)
    final_score_df = pd.read_csv(final_score_file)
    final_score_feature = final_score_df['feature']
    feature_df = data[final_score_feature]
    feature_name = feature_df.columns.tolist()
    pcc_matrix = np.corrcoef(feature_df, rowvar=False)
    
    ind = get_remained_feature_index(pcc_matrix, th)
    pcc_feature = [feature_name[i] for i in ind]
    remain_feature_num = len(pcc_feature)
    pcc_data = feature_df[pcc_feature]
    pcc_data_plus_label = pd.concat([data['label'],pcc_data],axis=1)
    return pcc_data_plus_label
    
pcc_data_plus_label = pcc_filter(sys.argv[1], sys.argv[2], float(sys.argv[3]))

pcc_data_plus_label.to_csv(sys.argv[4], index=False)

pcc_features = pcc_data_plus_label.columns.values.tolist()

pcc_test_data = pd.read_csv(sys.argv[5])

pcc_test = pcc_test_data[pcc_features]

pcc_test.to_csv(sys.argv[6], index=False)