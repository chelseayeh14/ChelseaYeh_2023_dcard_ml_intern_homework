import pandas as pd
from datetime import datetime
import pytz
from transformers import AutoTokenizer
import encoding as enc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def merge_crawler(df, df_crawler, col_post, col_title, col_create, col_like):
    ''' merge raw and crawler datasets '''
    df_crawler = df_crawler.dropna(how='all', subset=[col_post, col_create])
    df_crawler = df_crawler.drop_duplicates([col_title, col_create])
    df_crawler = df_crawler[[col_title, col_create, col_like]]
    df = df.merge(df_crawler, how='left', on=[col_title, col_create])

    return df


def chg_dtype(df, col_dt, col_cat):
    ''' change data types '''
    # change object to datetime
    df[col_dt]  = pd.to_datetime(df[col_dt]).dt.tz_convert(tz='ROC')
    # change object to category
    df[col_cat] = df[col_cat].astype('category')
    
    return df


def unif_forum_stats(df, col_id, col_stats, id):
    ''' uniform forum stats for each forum id '''
    ind = (df[col_id] == id)
    df.loc[ind, col_stats] = float(df.loc[ind, col_stats].mode())

    return df


def tokenizer(df, col_txt, len):
    ''' tokenize text column '''
    col_ids = ['token_{}'.format(str(i)) for i in range(len)]

    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-macbert-base')
    tokenized_data = tokenizer(list(df[col_txt]), return_tensors='np', max_length=len, padding='max_length', truncation=True)
    ids = tokenized_data['input_ids']
    df_ids = pd.DataFrame(ids, columns=col_ids)
    df = pd.concat([df_ids, df], axis=1)
    df.drop(['token_0', 'token_{}'.format(str(len-1)), col_txt], axis=1, inplace=True)

    return df


def add_create_days(df, now, col_create, col_days):
    ''' add created days '''
    now = datetime.now().replace(tzinfo=pytz.timezone('ROC'))
    interval = (now - df[col_create]).dt.days
    df[col_days] = interval

    return df


def break_create(df, col_dt, col_hr, col_wkd):
    ''' break created time into hour and weekday '''
    df[col_hr]  = df[col_dt].dt.hour.astype('category')
    df[col_wkd] = df[col_dt].dt.weekday.astype('category')
    df.drop(col_dt, axis=1, inplace=True)

    return df


def encode_cat(df, len_train, col, col_tgt, how='one-hot', stats='mean', q=0.5, n_splits=5, N_min=1000):
    ''' choose encoding method for categorical column '''
    df_train = df.iloc[:len_train].reset_index(drop=True)
    df_test  = df.iloc[len_train:].reset_index(drop=True)
    
    if   how == 'one-hot':
        df_train, df_test = enc.one_hot(df_train, df_test, col)
    elif how == 'frequency':
        df_train, df_test = enc.freq(df_train, df_test, col)
    elif how == 'target':
        df_train, df_test = enc.target(df_train, df_test, col, col_tgt, stats, q)
    elif how == 'k-fold beta target':
        df_train, df_test = enc.kfold_beta_target(df_train, df_test, col, col_tgt, n_splits, N_min)
    
    df = pd.concat([df_train, df_test], ignore_index=True)

    return df


def split_data(df, len_train, len_test_pub, len_test_pvt, col_tgt):
    ''' split datasets into feature and target variables '''
    last_train    = len_train
    last_test_pub = last_train + len_test_pub
    last_test_pvt = last_test_pub + len_test_pvt

    X_train    = df.iloc[:last_train].drop(col_tgt, axis=1)
    y_train    = df.iloc[:last_train][col_tgt].astype(int)
    X_test_pub = df.iloc[last_train:last_test_pub].drop(col_tgt, axis=1).reset_index(drop=True)
    y_test_pub = df.iloc[last_train:last_test_pub][col_tgt].reset_index(drop=True).astype(int)
    X_test_pvt = df.iloc[last_test_pub:last_test_pvt].drop(col_tgt, axis=1).reset_index(drop=True)

    return X_train, y_train, X_test_pub, y_test_pub, X_test_pvt


def split_data_subset(df, len_train, len_test_pub, len_test_pvt, col_tgt, col_drop, col_na, col_comp, num_comp, index):
    ''' split datasets with subsetting condition '''
    
    def subset_cond(df, col_na, col_comp, num_comp, index):
        ''' return indicators of subsetting conditions '''
        ind_na   = df[col_na].isna()
        ind_comp = df[col_comp] > num_comp
        
        if   index == 0:
            return  ind_na &  ind_comp
        elif index == 1:
            return  ind_na & ~ind_comp
        elif index == 2:
            return ~ind_na &  ind_comp
        elif index == 3:
            return ~ind_na & ~ind_comp

    
    last_train    = len_train
    last_test_pub = last_train + len_test_pub
    last_test_pvt = last_test_pub + len_test_pvt
    
    df_train    = df.iloc[:last_train]
    df_test_pub = df.iloc[last_train:last_test_pub]
    df_test_pvt = df.iloc[last_test_pub:last_test_pvt]

    # subset datasets
    df_train    = df_train[   subset_cond(df_train,    col_na, col_comp, num_comp, index)]
    df_test_pub = df_test_pub[subset_cond(df_test_pub, col_na, col_comp, num_comp, index)]
    df_test_pvt = df_test_pvt[subset_cond(df_test_pvt, col_na, col_comp, num_comp, index)]

    # drop create_days and like_count_now if like_count_now is missing
    if index in [0, 1]:
        df_train.drop(   col_drop, axis=1, inplace=True)
        df_test_pub.drop(col_drop, axis=1, inplace=True)
        df_test_pvt.drop(col_drop, axis=1, inplace=True)

    test_pvt_index = df_test_pvt.index
    X_train    = df_train.drop([col_tgt, col_comp], axis=1)
    y_train    = df_train[col_tgt].astype(int)
    X_test_pub = df_test_pub.drop([col_tgt, col_comp], axis=1).reset_index(drop=True)
    y_test_pub = df_test_pub[col_tgt].reset_index(drop=True).astype(int)
    X_test_pvt = df_test_pvt.drop([col_tgt, col_comp], axis=1).reset_index(drop=True)

    return X_train, y_train, X_test_pub, y_test_pub, X_test_pvt, test_pvt_index


def standardize(X_train, X_test_pub ,X_test_pvt):
    ''' use standardization to put numeircal values into same scale '''
    scaler = StandardScaler()

    X_train    = scaler.fit_transform(X_train)
    X_test_pub = scaler.transform(X_test_pub)
    X_test_pvt = scaler.transform(X_test_pvt)

    return X_train, X_test_pub, X_test_pvt


def pca(X_train, X_test_pub ,X_test_pvt, n_components='mle'):
    ''' use PCA to reduce dimensions '''
    p = PCA(n_components=n_components)

    X_train    = p.fit_transform(X_train)
    X_test_pub = p.transform(X_test_pub)
    X_test_pvt = p.transform(X_test_pvt)
    
    return X_train, X_test_pub, X_test_pvt