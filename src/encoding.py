import numpy as np
import pandas as pd
import kfold_beta_tgt_encoding as kbtenc

def one_hot(df_train, df_test, col):
    ''' use One-Hot Encoding for categorical column '''
    df_train = pd.get_dummies(df_train, prefix_sep='_', columns=[col])
    df_test  = pd.get_dummies(df_test,  prefix_sep='_', columns=[col])

    return df_train, df_test


def freq(df_train, df_test, col):
    ''' use Frequency Encoding for categorical column '''
    col_drop = f'{col}_drop'
    
    cnt = df_train.groupby(col)[col].count()
    df_train = df_train.merge(cnt, how='left', left_on=col, right_index=True, suffixes=('_drop', '')).drop(col_drop, axis=1)
    df_test  = df_test.merge( cnt, how='left', left_on=col, right_index=True, suffixes=('_drop', '')).drop(col_drop, axis=1)
    df_test[col] = df_test[col].replace(np.nan, 0)

    return df_train, df_test


def target(df_train, df_test, col, col_tgt, stats, q):
    ''' use Target Encoding for categorical column '''
    col_drop = f'{col_tgt}_drop'
    
    # calculate statistics
    if   stats == 'mean':
        tgt_stats = df_train.groupby(col)[col_tgt].mean()
    elif stats == 'median':
        tgt_stats = df_train.groupby(col)[col_tgt].median()
    elif stats == 'mode':
        tgt_stats = df_train.groupby(col)[col_tgt].mode()
    elif stats == 'quantile':
        tgt_stats = df_train.groupby(col)[col_tgt].quantile(q)
    elif stats == 'var':
        tgt_stats = df_train.groupby(col)[col_tgt].var()
    elif stats == 'std':
        tgt_stats = df_train.groupby(col)[col_tgt].std()

    df_train = df_train.merge(tgt_stats, how='left', left_on=col, right_index=True, suffixes=('', '_drop'))
    df_test  = df_test.merge( tgt_stats, how='left', left_on=col, right_index=True, suffixes=('', '_drop'))
    df_train.drop(col, axis=1, inplace=True)
    df_test.drop( col, axis=1, inplace=True)
    df_train.rename(columns={col_drop: col}, inplace=True)
    df_test.rename( columns={col_drop: col}, inplace=True)
    df_test[col] = df_test[col].replace(np.nan, 0)

    return df_train, df_test


def kfold_beta_target(df_train, df_test, col, col_tgt, n_splits, N_min):
    ''' use K-fold Beta Target Encoding for categorical column '''
    df_train, df_test = kbtenc.kfold_beta_target_encoding(df_train, df_test, col, col_tgt, n_splits, N_min)

    return df_train, df_test