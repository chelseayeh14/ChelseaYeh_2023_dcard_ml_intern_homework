import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

class BetaEncoder(object):
    ''' create BetaEncoder class '''
    def __init__(self, group):
        
        self.group = group
        self.stats = None
    
    def fit(self, df, target_col):
        ''' get sum and count of target column '''
        self.prior_mean = np.mean(df[target_col])
        stats = df[[target_col, self.group]].groupby(self.group)
        stats = stats.agg(['sum', 'count'])[target_col]
        stats.rename(columns={'sum': 'n', 'count': 'N'}, inplace=True)
        stats.reset_index(level=0, inplace=True)
        self.stats = stats
    
    def transform(self, df, stat_type, N_min=1):
        ''' extract posterior statistics '''
        df_stats = pd.merge(df[[self.group]], self.stats, how='left')
        n = df_stats['n'].copy()
        N = df_stats['N'].copy()
        
        # fill in missing values in count and sum
        nan_indexs = np.isnan(n)
        n[nan_indexs] = self.prior_mean
        N[nan_indexs] = 1.0
        
        # prior parameters
        N_prior     = np.maximum(N_min - N, 0)
        alpha_prior = self.prior_mean * N_prior
        beta_prior  = (1 - self.prior_mean) * N_prior
        
        # posterior parameters
        alpha = alpha_prior + n
        beta  = beta_prior + N - n
    
        # calculate statistics
        if   stat_type == 'mean':
            num = alpha
            dem = alpha + beta
        elif stat_type == 'mode':
            num = alpha - 1
            dem = alpha + beta - 2  
        elif stat_type == 'median':
            num = alpha - 1/3
            dem = alpha + beta - 2/3
        elif stat_type == 'var':
            num = alpha * beta
            dem = (alpha + beta)**2 * (alpha + beta + 1) 
        elif stat_type == 'skew':
            num = 2 * (beta - alpha) * np.sqrt(alpha + beta + 1)
            dem = (alpha + beta + 2) * np.sqrt(alpha * beta)
        elif stat_type == 'kurt':
            num = 6 * (alpha - beta)**2 * (alpha + beta + 1) - alpha * beta * (alpha + beta + 2)
            dem = alpha * beta * (alpha + beta + 2) * (alpha + beta + 3)

        # fill in missing values in statistics
        value = num / dem
        value[np.isnan(value)] = np.nanmedian(value)
        
        return value
    

def kfold_beta_target_encoding(df_train, df_test, col, col_tgt, n_splits, N_min):
    ''' use K-Fold Beta Target Encoding for categorical column '''
    # set up statistics
    stats_col = {
        'mean':   f'{col}_mean',
        'mode':   f'{col}_mode',
        'median': f'{col}_median',
        'var':    f'{col}_var',
        'skew':   f'{col}_skew',
        'kurt':   f'{col}_kurt'
    }

    # encode training data
    be_train = BetaEncoder(col)

    ## split dataset into k folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=2)
    for kf_train, kf_test in kf.split(df_train):
        be_train.fit(df_train.iloc[kf_train], col_tgt)
        for stat, stat_col in stats_col.items():
            tgt_stats = be_train.transform(df_train.iloc[kf_test], stat, N_min).set_axis(kf_test)
            df_train.loc[kf_test, stat_col] = tgt_stats
        
    # encode test data
    be_test = BetaEncoder(col)
    be_test.fit(df_train, col_tgt)
    for stat, stat_col in stats_col.items():
        tgt_stats = be_test.transform(df_test, stat, N_min).set_axis(df_test.index)
        df_test[stat_col] = tgt_stats
    
    # drop statistics columns with any missing value
    for stat_col in stats_col.values():
        ind_na = (df_train[stat_col].isna().sum() != 0) | (df_test[stat_col].isna().sum() != 0)
        if ind_na:
            df_train.drop(stat_col, axis=1, inplace=True)
            df_test.drop( stat_col, axis=1, inplace=True)
    
    # drop categorical column
    df_train.drop(col, axis=1, inplace=True)
    df_test.drop( col, axis=1, inplace=True)

    return df_train, df_test