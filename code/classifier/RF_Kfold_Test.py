import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import minmax_scale

def rf_kfold(X, y, output_path, k_fold_cv, n_estimators, method_name):
    
    output_path = output_path
    result_dict = {}
    result_dict['method'] = method_name
    skf = KFold(n_splits = k_fold_cv, random_state = 123, shuffle = True)

    rf_model = RandomForestClassifier(n_estimators = n_estimators, n_jobs = 4, random_state = 123)

    y_test_array = np.array([])
    y_pred_array = np.array([])
    y_pred_proba_array = np.array([])

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        y_pred_proba_1 = rf_model.predict_proba(X_test)[:,1]

        y_test_array = np.concatenate((y_test_array, y_test))
        y_pred_array = np.concatenate((y_pred_array, y_pred))
        y_pred_proba_array = np.concatenate((y_pred_proba_array, y_pred_proba_1))

    tn, fp, fn, tp = confusion_matrix(y_test_array, y_pred_array).ravel()

    Sn = tp / (tp + fn)

    Sp = tn / (tn + fp)

    result_dict['ACC'] = accuracy_score(y_test_array, y_pred_array)

    result_dict['MCC'] = matthews_corrcoef(y_test_array, y_pred_array)

    result_dict['AUC'] = roc_auc_score(y_test_array, y_pred_proba_array)

    result_dict['Sensitivity'] = Sn

    result_dict['Specificity'] = Sp

    original_data = pd.read_excel(output_path)

    result_df = pd.DataFrame(result_dict, index=[0])

    save_result = pd.concat([original_data, result_df], axis=0)

    save_result.to_excel(output_path, index=False)

def rf_independent_test(X_train, y_train, X_test, y_test, output_path, n_estimators, method_name):
    
    output_path = output_path
    result_dict = {}
    result_dict['method'] = method_name

    rf_model = RandomForestClassifier(n_estimators=n_estimators, n_jobs = 4, random_state=123)

    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    Sn = tp / (tp + fn)

    Sp = tn / (tn + fp)

    y_pred_proba = rf_model.predict_proba(X_test)

    result_dict['ACC'] = accuracy_score(y_test, y_pred)

    result_dict['MCC'] = matthews_corrcoef(y_test, y_pred)

    result_dict['AUC'] = roc_auc_score(y_test, y_pred_proba[:,1])

    result_dict['Sensitivity'] = Sn

    result_dict['Specificity'] = Sp

    original_data = pd.read_excel(output_path)

    result_df = pd.DataFrame(result_dict, index=[0])

    save_result = pd.concat([original_data, result_df], axis=0)

    save_result.to_excel(output_path, index=False)

train_data = pd.read_csv("../S-nitrosylation/cwgan_augmented_data/cwgan_augmented_data_pcc07.csv")

test_data = pd.read_csv("../S-nitrosylation/train_test/test_nitro_pcc07.csv")

def normalize_data(data):
    data = minmax_scale(data,axis=0)
    return data

train_data = train_data.values
# test_data = test_data.values
train_X = train_data[:,:-1]
train_y = train_data[:,-1]
# test_X = test_data[:,1:]
# test_y = test_data[:,0]
# test_X = normalize_data(test_X)

rf_kfold(train_X, train_y, "../kfold_result.xlsx", 10, 100, 'PFR5000_pcc07_CWGAN_RF_10-fold')

rf_independent_test(train_X, train_y, test_X, test_y, "../independent_test_result.xlsx", 100, "PFR5000_pcc07_CWGAN_RF")