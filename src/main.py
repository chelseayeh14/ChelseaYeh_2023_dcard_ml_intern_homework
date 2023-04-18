import os
import pandas as pd
import preprocessing as preproc
from datetime import datetime
import modeling as mdl

# import datasets
path = os.path.dirname(__file__)
path = os.path.join(path, '..')
path_train = os.path.join(path, 'dataset', 'intern_homework_train_dataset.csv')
path_test_pub = os.path.join(
    path, 'dataset', 'intern_homework_public_test_dataset.csv')
path_test_pvt = os.path.join(
    path, 'dataset', 'intern_homework_private_test_dataset.csv')
path_train_crawler = os.path.join(path, 'dataset', 'crawler_train_dataset.csv')
path_test_pub_crawler = os.path.join(
    path, 'dataset', 'crawler_public_test_dataset.csv')
path_test_pvt_crawler = os.path.join(
    path, 'dataset', 'crawler_private_test_dataset.csv')
path_pred_pvt = os.path.join(path, 'dataset', 'result.csv')

df_train = pd.read_csv(path_train,            sep=',', encoding='UTF-8')
df_test_pub = pd.read_csv(path_test_pub,         sep=',', encoding='UTF-8')
df_test_pvt = pd.read_csv(path_test_pvt,         sep=',', encoding='UTF-8')
df_train_crawler = pd.read_csv(
    path_train_crawler,    sep=',', encoding='UTF-8')
df_test_pub_crawler = pd.read_csv(
    path_test_pub_crawler, sep=',', encoding='UTF-8')
df_test_pvt_crawler = pd.read_csv(
    path_test_pvt_crawler, sep=',', encoding='UTF-8')
df = pd.concat([df_train, df_test_pub, df_test_pvt],
               join='outer', ignore_index=True)
df_crawler = pd.concat([df_train_crawler, df_test_pub_crawler,
                       df_test_pvt_crawler], join='outer', ignore_index=True)

# set up variables
len_train = 50000
len_test_pub = 10000
len_test_pvt = 10000
cols = [
    'like_count_1h',
    'like_count_2h',
    'like_count_3h',
    'like_count_4h',
    'like_count_5h',
    'like_count_6h',
    'like_count_now',
    'created_hr',
    'created_days',
    'forum_id',
    'like_count_24h'
]
col_ind = 'like_count_6h'
col_ind_now = 'like_count_now'
col_tgt = 'like_count_24h'
test_pvt_index = []
result = {
    'MLR':     [],
    'KNN':     [],
    'SVR':     [],
    'RFR':     [],
    'XGBoost': [],
    'LSTM':    []
}

# preprocessing
df = preproc.merge_crawler(df, df_crawler, 'post_id',
                           'title', 'created_at', 'like_count_now')
df = preproc.chg_dtype(df, col_dt='created_at', col_cat='forum_id')
df = preproc.add_create_days(df, datetime(
    2023, 4, 10, 0, 0, 0), col_create='created_at', col_days='created_days')
df = preproc.break_create(df, col_dt='created_at',
                          col_hr='created_hr', col_wkd='created_wkd')
df = df[cols]
df = preproc.encode_cat(df, len_train, col='created_hr',
                        col_tgt=col_tgt, how='target', stats='quantile', q=0.8)
df = preproc.encode_cat(df, len_train, col='forum_id',
                        col_tgt=col_tgt, how='frequency')

for i in range(4):
    split_result = preproc.split_data_subset(df, len_train, len_test_pub, len_test_pvt, col_tgt, col_drop=['created_days', 'like_count_now'], col_na='like_count_now',
                                             col_comp='forum_id', num_comp=3, index=i)
    X_train_raw, y_train, X_test_pub_raw, y_test_pub, X_test_pvt_raw = split_result[:5]
    test_pvt_index.extend(split_result[5])
    X_train, X_test_pub, X_test_pvt = preproc.standardize(
        X_train_raw, X_test_pub_raw, X_test_pvt_raw)

    # modeling
    pub_ind = X_test_pub_raw[col_ind]
    pvt_ind = X_test_pvt_raw[col_ind]
    now = False
    pub_ind_now = None
    pvt_ind_now = None
    if i in [2, 3]:
        now = True
        pub_ind_now = X_test_pub_raw[col_ind_now]
        pvt_ind_now = X_test_pvt_raw[col_ind_now]

    result['MLR'].append(mdl.multi_linear_reg(X_train, y_train, X_test_pub, y_test_pub, X_test_pvt, pub_ind, pvt_ind, now=now, pub_ind_now=pub_ind_now,
                                              pvt_ind_now=pvt_ind_now, train=True, test=True, pred=False))
    result['KNN'].append(mdl.knearest_neighbors(X_train, y_train, X_test_pub, y_test_pub, X_test_pvt, pub_ind, pvt_ind, now=now, pub_ind_now=pub_ind_now,
                                                pvt_ind_now=pvt_ind_now, train=True, test=True, pred=False, n_neighbors=3, weights='distance'))
    result['SVR'].append(mdl.sup_vec_reg(X_train, y_train, X_test_pub, y_test_pub, X_test_pvt, pub_ind, pvt_ind, now=now, pub_ind_now=pub_ind_now,
                                         pvt_ind_now=pvt_ind_now, train=True, test=True, pred=True))
    result['RFR'].append(mdl.random_forest_reg(X_train, y_train, X_test_pub, y_test_pub, X_test_pvt, pub_ind, pvt_ind, now=now, pub_ind_now=pub_ind_now,
                                               pvt_ind_now=pvt_ind_now, train=True, test=True, pred=False, n_estimators=200, random_state=42))
    result['XGBoost'].append(mdl.xgboost(X_train, y_train, X_test_pub, y_test_pub, X_test_pvt, pub_ind, pvt_ind, now=now, pub_ind_now=pub_ind_now,
                                         pvt_ind_now=pvt_ind_now, train=True, test=True, pred=False, colsample_bytree=0.7, learning_rate=0.1, max_depth=5,
                                         n_estimators=1000, objective='reg:squarederror'))

mdl.show_result(result, path_pred_pvt, col_tgt,
                multi=True, index=test_pvt_index)
