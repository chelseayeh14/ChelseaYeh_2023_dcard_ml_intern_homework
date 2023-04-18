import os
import pandas as pd
import preprocessing as preproc
import modeling as mdl

# import datasets
path = os.path.dirname(__file__)
path = os.path.join(path, '..')
path_train = os.path.join(path, 'dataset', 'intern_homework_train_dataset.csv')
path_test_pub = os.path.join(
    path, 'dataset', 'intern_homework_public_test_dataset.csv')
path_test_pvt = os.path.join(
    path, 'dataset', 'intern_homework_private_test_dataset.csv')
path_pred_pvt = os.path.join(path, 'dataset', 'result.csv')

df_train = pd.read_csv(path_train,    sep=',', encoding='UTF-8')
df_test_pub = pd.read_csv(path_test_pub, sep=',', encoding='UTF-8')
df_test_pvt = pd.read_csv(path_test_pvt, sep=',', encoding='UTF-8')
df = pd.concat([df_train, df_test_pub, df_test_pvt],
               join='outer', ignore_index=True)

# set up variables
len_train = 50000
len_test_pub = 10000
len_test_pvt = 10000
col_ind = 'like_count_6h'
col_tgt = 'like_count_24h'
result = {
    'MLR':     [],
    'KNN':     [],
    'SVR':     [],
    'RFR':     [],
    'XGBoost': [],
    'LSTM':    []
}

# preprocessing
df = preproc.chg_dtype(df, col_dt='created_at', col_cat=[
                       'forum_id', 'author_id'])
df_unif = preproc.unif_forum_stats(
    df.iloc[:len_train], 'forum_id', 'forum_stats', 47568)
df = pd.concat([df_unif, df[len_train:]], ignore_index=True)
df = preproc.tokenizer(df, col_txt='title', len=7)
df = preproc.break_create(df, col_dt='created_at',
                          col_hr='created_hr', col_wkd='created_wkd')
df = preproc.encode_cat(df, len_train, col='created_hr',
                        col_tgt=col_tgt, how='frequency')
df = preproc.encode_cat(df, len_train, col='created_wkd',
                        col_tgt=col_tgt, how='frequency')
df = preproc.encode_cat(df, len_train, col='forum_id',
                        col_tgt=col_tgt, how='k-fold beta target', n_splits=5, N_min=3)
df = preproc.encode_cat(df, len_train, col='author_id',
                        col_tgt=col_tgt, how='k-fold beta target', n_splits=5, N_min=3)

X_train_raw, y_train, X_test_pub_raw, y_test_pub, X_test_pvt_raw = preproc.split_data(
    df, len_train, len_test_pub, len_test_pvt, col_tgt)
X_train, X_test_pub, X_test_pvt = preproc.standardize(
    X_train_raw, X_test_pub_raw, X_test_pvt_raw)
X_train, X_test_pub, X_test_pvt = preproc.pca(
    X_train, X_test_pub, X_test_pvt, n_components=0.95)

# modeling
pub_ind = X_test_pub_raw[col_ind]
pvt_ind = X_test_pvt_raw[col_ind]

result['MLR'].append(mdl.multi_linear_reg(X_train, y_train, X_test_pub,
                     y_test_pub, X_test_pvt, pub_ind, pvt_ind, train=True, test=True, pred=False))
result['KNN'].append(mdl.knearest_neighbors(X_train, y_train, X_test_pub, y_test_pub, X_test_pvt, pub_ind, pvt_ind, train=True, test=True, pred=False,
                                            n_neighbors=3, weights='distance'))
result['SVR'].append(mdl.sup_vec_reg(X_train, y_train, X_test_pub, y_test_pub,
                     X_test_pvt, pub_ind, pvt_ind, train=True, test=True, pred=True))
result['RFR'].append(mdl.random_forest_reg(X_train, y_train, X_test_pub, y_test_pub, X_test_pvt, pub_ind, pvt_ind, train=True, test=True, pred=False,
                                           n_estimators=200, random_state=42))
result['XGBoost'].append(mdl.xgboost(X_train, y_train, X_test_pub, y_test_pub, X_test_pvt, pub_ind, pvt_ind, train=True, test=True, pred=False,
                                     colsample_bytree=0.7, learning_rate=0.1, max_depth=5, n_estimators=1000, objective='reg:squarederror'))

mdl.show_result(result, path_pred_pvt, col_tgt, multi=False)
