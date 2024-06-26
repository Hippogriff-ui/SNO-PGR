import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

# 读取原始数据
original_data = pd.read_csv('../S-nitrosylation/train_test/Nitro_train_pcc07.csv')

train_data = original_data.values
X = train_data[:,1:]
y = train_data[:,0]

# 使用SMOTE进行数据增强
smote = SMOTE(random_state=123)
X_resampled, y_resampled = smote.fit_resample(X, y)
y_resampled_1 = y_resampled.reshape(-1,1)
resampled_data = np.hstack((y_resampled_1, X_resampled))
resampled_data_df = pd.DataFrame(resampled_data)
resampled_data_df.to_csv(r'D:\paper\S-nitrosylation\train_test\Nitro_train_pcc07_SMOTE.csv', index=False)