import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

def model(mdl, X_train, y_train, X_test_pub, y_test_pub, X_test_pvt, pub_ind, pvt_ind, now, pub_ind_now, pvt_ind_now, train, test, pred):
    ''' train/test/predict data '''
    
    def calc_mape(y_true, y_pred):
        ''' calculate MAPE between actual and predicted values '''
        y_true, y_pred = np.array(y_true), np.array(y_pred)

        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    

    def train(mdl, X_train, y_train):
        ''' train data '''
        mdl.fit(X_train, y_train)


    def test(mdl, X_test, y_test, ind, now, ind_now):
        ''' test data '''

        y_pred = mdl.predict(X_test)
        y_pred = np.where(y_pred >= ind, y_pred, ind)
        if now:
            y_pred = np.where(y_pred <= ind_now, y_pred, ind_now)
        mape = round(calc_mape(y_test, y_pred), ndigits=2)

        return mape


    def pred(mdl, X_test, ind, now, ind_now):
        ''' predict data '''

        y_pred = mdl.predict(X_test)
        y_pred = np.where(y_pred >= ind, y_pred, ind)
        if now:
            y_pred = np.where(y_pred <= ind_now, y_pred, ind_now)

        return y_pred


    mape   = None
    y_pred = None

    if train:
        train(mdl, X_train, y_train)
    
    if test:
        mape = test(mdl, X_test_pub, y_test_pub, pub_ind, now, pub_ind_now)
    
    if pred:
        y_pred = pred(mdl, X_test_pvt, pvt_ind, now, pvt_ind_now)
    
    return mape, y_pred


def multi_linear_reg(X_train, y_train, X_test_pub, y_test_pub, X_test_pvt, pub_ind, pvt_ind, now=False, pub_ind_now=None, pvt_ind_now=None,
                     train=True, test=False, pred=False):
    ''' Multiple Linear Regression '''
    mlr = LinearRegression()

    return model(mlr, X_train, y_train, X_test_pub, y_test_pub, X_test_pvt, pub_ind, pvt_ind, now, pub_ind_now, pvt_ind_now, train, test, pred)


def knearest_neighbors(X_train, y_train, X_test_pub, y_test_pub, X_test_pvt, pub_ind, pvt_ind, now=False, pub_ind_now=None, pvt_ind_now=None,
                       train=True, test=False, pred=False, n_neighbors=3, weights='distance'):
    ''' K-Nearest Neighbors '''
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)

    return model(knn, X_train, y_train, X_test_pub, y_test_pub, X_test_pvt, pub_ind, pvt_ind, now, pub_ind_now, pvt_ind_now, train, test, pred)


def sup_vec_reg(X_train, y_train, X_test_pub, y_test_pub, X_test_pvt, pub_ind, pvt_ind, now=False, pub_ind_now=None, pvt_ind_now=None,
                train=True, test=False, pred=False):
    ''' Support Vector Regression '''
    svr = SVR()

    return model(svr, X_train, y_train, X_test_pub, y_test_pub, X_test_pvt, pub_ind, pvt_ind, now, pub_ind_now, pvt_ind_now, train, test, pred)


def random_forest_reg(X_train, y_train, X_test_pub, y_test_pub, X_test_pvt, pub_ind, pvt_ind, now=False, pub_ind_now=None, pvt_ind_now=None,
                      train=True, test=False, pred=False, n_estimators=200, random_state=42):
    ''' Random Forest Regression '''
    rfr = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    return model(rfr, X_train, y_train, X_test_pub, y_test_pub, X_test_pvt, pub_ind, pvt_ind, now, pub_ind_now, pvt_ind_now, train, test, pred)


def xgboost(X_train, y_train, X_test_pub, y_test_pub, X_test_pvt, pub_ind, pvt_ind, now=False, pub_ind_now=None, pvt_ind_now=None,
            train=True, test=False, pred=False, colsample_bytree=0.7, learning_rate=0.1, max_depth=5, n_estimators=1000, objective='reg:squarederror'):
    ''' eXtreme Gradient Boosting '''
    xgb = XGBRegressor(colsample_bytree=colsample_bytree, learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators,
                       objective=objective)
    
    return model(xgb, X_train, y_train, X_test_pub, y_test_pub, X_test_pvt, pub_ind, pvt_ind, now, pub_ind_now, pvt_ind_now, train, test, pred)


def show_result(result, path, col_tgt, multi=False, index=None):
    ''' print MAPE and export predicted value'''
    
    def print_mape(model, model_result):
        ''' print MAPE of each model '''
        for i in range(len(model_result)):
            mape = model_result[i][0]
            if mape is not None:
                print(f'{model} ({i+1}): {mape}%')


    def export_pred(model_result, col_tgt, multi, index):
        ''' export predicted value from selected model '''
        # check if datasets are subsetted
        if not multi:
            if len(model_result):
                y_pred_pvt = np.round(model_result[0][1], 0).astype(int)

                df_y_pred_pvt = pd.DataFrame({col_tgt: y_pred_pvt})
                df_y_pred_pvt.to_csv(path, index=False)
        else:
            y_pred_pvt_list = []
            if len(model_result):
                for i in range(len(model_result)):
                    # link predicted values from different conditions
                    y_pred_pvt_list.extend(model_result[i][1])
                
                y_pred_pvt_list = np.round(y_pred_pvt_list, 0).astype(int)
                df_y_pred_pvt = pd.DataFrame({col_tgt: y_pred_pvt_list}, index=index)
                df_y_pred_pvt.sort_index(inplace=True)
                df_y_pred_pvt.to_csv(path, index=False)
                
    for model, model_result in result.items():
        print_mape(model, model_result)
        export_pred(model_result, col_tgt, multi, index)