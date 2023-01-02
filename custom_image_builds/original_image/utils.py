# Databricks notebook source
import pandas as pd
import numpy as np

from collections import Counter

from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import Dense, Activation, Dropout
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.models import load_model

from pytz import timezone, utc
from datetime import timedelta, datetime
import datetime
import time
import logging
import os
import json
from joblib import dump, load
import pickle
from google.cloud import storage

from sklearn.naive_bayes import GaussianNB
from scipy.special import logsumexp


# import tensorflow as tf
# assert tf.__version__=='2.3.0'

from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

import mlflow.pyfunc



np.seterr(all='raise')

eligible_clubs= (4738,4787,6309,6310,6311,6312,6321,6324,6436,6535,8149,8164,
        8167,8183,8185, 4724,4776,4817,4836,4989,4990,6329,6361,6435,8102,8106,
        8107,8119,8196, 8213,8236,8247,4969,4808,6449)

eligible_cats= (19, 28, 55, 22, 23, 33, 34, 95, 8, 41, 43, 46, 49, 51, 40, 52, 1, 58, 78, 5,
          6, 20, 29, 31, 64, 61, 70, 83, 10, 14, 15, 16, 7, 11, 12, 17, 18, 21,
          60, 89, 3, 4, 13, 53, 94, 98, 66, 67, 68, 2, 27, 47, 54, 36, 9, 86)

eligible_cats_PI= (19, 28, 55, 22, 23, 33, 34, 95, 8, 41, 43, 46, 49, 51, 40, 52, 1, 58, 78, 5,
          6, 20, 29, 31, 64, 61, 70, 83, 10, 14, 15, 16, 7, 11, 12, 17, 18, 21,
          60, 89, 3, 4, 13, 53, 94, 98, 66, 67, 68, 2, 27, 47, 54, 36, 9, 86,
          37, 38, 39, 42, 44, 48, 56, 57, 72, 93, 96)

CONFIG_DIR = "/dbfs/OYI/prod_artifacts/"

"""
The below classes override TransformerMixin to create transformers either equivalent to
their scikit-learn preprocessing/pipeline counter parts but are dataframe aware
And all these functions have no side-effects, ie. don't modify the original dataframe
"""

class DataFrameSelector(BaseEstimator, TransformerMixin):
    '''
    Select columns from pandas dataframe by specifying a list of column names
    From this source:
    https://github.com/philipmgoddard/pipelines/blob/master/custom_transformers.py
    '''
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.loc[:, self.attribute_names].copy()


class LogFeaturizer(BaseEstimator,TransformerMixin):
    '''
    Log1p transforms inputs, filling NAs with zeroes
    '''
    def fit(self, X,y=None):
        return self

    def transform(self,X):
        X[X < 0] = 0
        res= np.log1p(X.fillna(0)).values
        return pd.DataFrame(res, columns= [i+'_log' for i in X.columns])

class ClipFeaturizer(BaseEstimator,TransformerMixin):
    '''
    Clips input values below min_value to min_value, and/or
    max_value to max value.
    '''
    def __init__(self, min_value=None, max_value=None):
        self.min_value=min_value
        self.max_value=max_value

    def fit(self, X,y=None):
        return self

    def transform(self,X):
        X= X.copy()
        if self.min_value is not None:
            X[X< self.min_value]= self.min_value
        if self.max_value is not None:
            X[X>self.max_value]= self.max_value
        return X

# LocationExtractor is used in cancelled items ml model 
class LocationExtractor(BaseEstimator,TransformerMixin):
    '''
    Extracts location from the Reserve and Sales_Floor_Location column
    Currently does not enforce the columns to be strings
    Also untested when providing just one column (might break if DataFrameSelector
    on a single column returns a Series instead of a DataFrame)
    '''
    def fit(self, X,y=None):
        return self

    def transform(self,X):
        newdf= {}
        for col in X.columns:
            extract= X.loc[:,col].str.extract('(\w+)-.*').iloc[:,0]
            extract= extract.fillna(value='na_{0}'.format(col))
            newdf[col+'_proc']= extract
        return pd.DataFrame(newdf)

class TimeExtractor(BaseEstimator,TransformerMixin):
    '''
    converts date time to weekday and adds it as a new column
    '''
    def fit(self, X,y=None):
        return self

    def transform(self,X):
        newdf= {}
        for col in X.columns:
            extract= X.loc[:,col].apply(lambda x: x.weekday())
            newdf[col+'_proc']= extract
        return pd.DataFrame(newdf)

class CategoryFeaturizer(BaseEstimator,TransformerMixin):
    '''
    Returns Dummy variables of categorical inputs (assumes that they are categorical for now)
    Accepts strings and integers
    Important: Will work even if the testing dataset that the object is transforming has fewer
    categories than the fitted dataset, and so will have the same number of columns as the latter
    '''
    def __init__(self):
        self.onehot_enc= OneHotEncoder(sparse=False,dtype='int', handle_unknown='ignore')

    def fit(self, X,y=None):
        self.onehot_enc.fit(X)
        self.colnames=[]
        for i,col in enumerate(X.columns):
            for level in self.onehot_enc.categories_[i]:
                self.colnames.append(col+'_'+str(level))
        return self

    def transform(self,X):
        res= self.onehot_enc.transform(X)
        return pd.DataFrame(res, columns= self.colnames)


class ColumnMerge(BaseEstimator, TransformerMixin):
    '''
    Like scikit-learn's FeatureUnion but dataframe aware
    '''
    def __init__(self,transformer_list, n_jobs=None, transformer_weights=None):
        self.tf_list= transformer_list

    def fit(self, X, y=None):
        for tf_name,tf in self.tf_list:
            tf.fit(X)
        return self

    def transform(self, X):
        res=[]
        for tf_name,tf in self.tf_list:
            res.append(tf.transform(X).reset_index(drop=True))
        res= pd.concat(res, axis=1)
        return res

class ModelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        self.model.fit(X)
        return self

    def transform(self, X, **transform_params):
        df =  pd.DataFrame(self.model.predict(X), columns=['result']).reset_index(drop=True)
        df.index = list(df.index)
        return df


class Stage1_NeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, input_shape=None, num_classes=5, batch_size=128, epochs=20, verbose=2):
        self.num_classes=num_classes
        self.batch_size=batch_size
        self.epochs=epochs
        self.verbose=verbose
        self.input_shape = input_shape

    def fit(self, X, y):
        # prepare data for keras-NN
        x_train= X.values
        self.classes_ = np.unique(y)
        y_train= pd.get_dummies(y).values

        # define model
        model = Sequential()
        model.add(Dense(100, activation='relu', input_shape=(x_train.shape[1],), kernel_initializer='glorot_uniform', bias_initializer='zeros'))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'))
        model.add(Dropout(0.2))
        model.add(Dense(25, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'))
        model.add(Dropout(0.2))
        model.add(Dense(12, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'))
        model.add(Dropout(0.2))
        model.add(Dense(9, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='softmax', kernel_initializer='glorot_uniform', bias_initializer='zeros'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(),  metrics=['accuracy'])
        self.model = model
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose)
        return self

    def predict_proba(self, X):
        x_test= X.values
        return self.model.predict(x_test)

    def save(self, path):
        self.model.save(path)


def stage2_nn(input_dimen=45):
    model = Sequential()
    model.add(Dense(100, input_dim=input_dimen,  activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(70, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

class EnsembleClassifierWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, X):
        return self.model.predict_proba(X)

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, stage1_preprocessor, stage1_classifiers, stage2_preprocessor, stage2_classifier):
        self.stage1_classifiers = stage1_classifiers
        self.stage1_preprocessor = stage1_preprocessor
        self.stage2_preprocessor = stage2_preprocessor
        self.stage2_classifier = stage2_classifier
        self.classes_ = None

    def _prepare_stage2_data(self, X, y):
        X_transformed = self.stage1_preprocessor.transform(X)
        s2_x = []
        for clf in self.stage1_classifiers:
            s2_x.append(clf.predict_proba(X_transformed))
        s2_x = np.hstack(s2_x)

        X_transformed2 = self.stage2_preprocessor.fit_transform(X)
        s2_x = np.hstack([s2_x, X_transformed2.values])
        s2_y= y.values
        return s2_x, s2_y

    def fit(self, X, y, val_x=None, val_y=None):
        self.classes_ = np.unique(y)
        s2_train_x, s2_train_y = self._prepare_stage2_data(X, y)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        self.stage2_classifier.fit(s2_train_x, s2_train_y, validation_split=0.2, callbacks=[es, mc])
        self.stage2_classifier = load_model('best_model.h5')
        return self

    def predict_proba(self, X):
        # Use self.s1_models to get s2 test-data
        X_transformed = self.stage1_preprocessor.transform(X)
        s2_x = []
        for clf in self.stage1_classifiers:
            s2_x.append(clf.predict_proba(X_transformed))
        s2_test_x = np.hstack(s2_x)
        X_transformed2 = self.stage2_preprocessor.transform(X)
        s2_test_x = np.hstack([s2_test_x, X_transformed2.values])
        return self.stage2_classifier.predict_proba(s2_test_x)

class MinMaxScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.min_max_scalar= MinMaxScaler()

    def fit(self, X, y=None):
        self.min_max_scalar.fit(X)
        return self

    def transform(self, X):
        arr = self.min_max_scalar.transform(X)
        return pd.DataFrame(arr, columns=list(X.columns))


# Functions for tpr-calculations

# input: pandas.core.series.Series Eg: ( [2,4,6,7,8], [2,5,1,8,9], [1,15,3,5,2], [2,7,10,9,1] )
# output: [7, 31, 20, 29, 20]
def _sum_value_counts(pd_series):
    return np.sum(list(pd_series), axis=0).tolist()


# input: pandas.core.series.Series Eg: ['No Action Taken, already out for sale', 'Add to picklist', 'Add to picklist', 'Location updated for the item',....]
# output: [c1, c2, c3, c4, c5] where (c1-c5) are the value_counts of target-values in the series
def _value_counts(pd_series):
    arr = []
    target_values = ['No Action Taken, already out for sale', 'Add to picklist', 'Updated the on hands quantity for the item', 'Location updated for the item',
                     'New price print sign has been printed']
    c = Counter(pd_series)
    for v in target_values:
        arr.append(c[v])
    return arr


def calculate_daily_tpr(df, level, path='', save=False):
    level_name = '_'.join(_ for _ in level)
    groupbycol = level.copy()
    groupbycol.append('run_date')

    grouped = df.groupby(groupbycol).agg({'log_id':'count','event_note': lambda x:_value_counts(x) }).\
            reset_index().rename(columns={'log_id':'total_action_cnt','event_note':'action_distr'})

    joined = grouped.merge(grouped,on=level,how='left',suffixes=('','_1'))

    joined = joined.loc[joined['run_date']>joined['run_date_1']]

    joined = joined.groupby(groupbycol).agg({'total_action_cnt_1':'sum','action_distr_1': lambda x:_sum_value_counts(x) }).\
        reset_index().rename(columns={'total_action_cnt_1':'total_action_cnt','action_distr_1':'action_distr'})

    joined[level_name + '_tpr'] = joined.apply(lambda row: [round(x/row['total_action_cnt'], 3) for x in row['action_distr']], axis=1)
    joined.drop(columns=['action_distr', 'total_action_cnt'], inplace=True)

    target_values = ['no_action_taken', 'add_to_picklist', 'update_ohq', 'update_loc', 'new_price_sign']
    target_cols = [level_name + "_" + x + "_tpr" for x in target_values]
    joined[target_cols] = pd.DataFrame(joined[level_name + '_tpr'].values.tolist(), index= joined.index)

    select_col = groupbycol.copy()
    select_col += target_cols

    if save:
        to_dump = joined.loc[joined['run_date']==joined.run_date.max()]
        to_dump = to_dump[select_col]
        local_path = level_name + '_tpr.joblib'
        dump(to_dump, local_path)
        storage_path = os.path.join(path, local_path)
        blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
        blob.upload_from_filename(local_path)
        

    df = df.merge(joined[select_col], on=groupbycol, how='left')
    return df


def calculate_all_level_tpr(df, path='', save=False):
    levels = [ ['mkt'], ['reg'], ['club_nbr'], ['cat'], ['item_nbr'], ['club_nbr','cat'], ['state','cat'], ['mkt','cat'], ['reg','cat'] ]
    for level in levels:
        df = calculate_daily_tpr(df, level, path, save)
    return df



def load_tpr_features(nosales_test, path, config=None):
    df = nosales_test.copy()
    levels = [ ['mkt'], ['reg'], ['club_nbr'], ['cat'], ['item_nbr'],
              ['club_nbr','cat'], ['state','cat'], ['mkt','cat'], ['reg','cat'] ]

    for level in levels:
        level_name = '_'.join(_ for _ in level)
        file_name = level_name + '_tpr.joblib'

        if config is None:
            tmp = load(os.path.join(path, file_name)) #not used in production
        else:
            tmp = load("{config['model_path']}/{0}".format(file_name))

        tmp = tmp.drop('run_date', axis=1)
        df = df.merge(tmp, on=level, how='left')

    return df



def get_raw_score_thresholds(train):
    club_thresh = {}

    mins, maxs= {},{}

    for club in train.club_nbr.unique():
        train_club = train[train.club_nbr==club]
        thresholds = np.sort(list(set(np.round(train_club.raw_score.unique(), 4))))

        f1_arr = []
        prec_arr = []
        recall_arr= []
        for th in thresholds:
            y_pred = list(train_club.raw_score >= th)
            y_true = list(train_club.action == True)
            f1 = f1_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1_arr.append(f1)
            prec_arr.append(prec)
            recall_arr.append(recall)

        club_thresh[club] = thresholds[np.argmax(f1_arr)]

    return club_thresh



def gen_thresholds(df, predictions, classes):
    cutoff = (df.run_date.max() - timedelta(days=7)).strftime('%Y-%m-%d')
    df = df.reset_index(drop=True)
    scores = pd.DataFrame(predictions, columns=classes)

    action_cols= ['Add to picklist', 'Location updated for the item',
                  'New price print sign has been printed', 'Updated the on hands quantity for the item']
    scores['total_score'] = scores.loc[:,action_cols].sum(axis=1)

    # 'Location updated for the item', 'New price print sign has been printed'
    df['act_bool']= df.event_note.isin(action_cols)*1
    df['raw_score'] = scores['total_score']
    df['action']= ~(df.event_note.isin(['No Action Taken, already out for sale','No Action Taken, already OFS']))

    cols = ["central_dt", "club_nbr", "item_nbr", "event_note", "action", "act_bool", "run_date", "old_nbr", "raw_score"]
    df = df[cols]
    df_subset = df[pd.to_datetime(df.central_dt) >= cutoff]
    np.sort(df_subset.central_dt.unique())
    thresh = get_raw_score_thresholds(df_subset)
    return thresh


def get_config(mode=1):

    config_file = os.path.join(CONFIG_DIR,'config.json')

    with open(config_file,'r') as f:
        config= json.load(f)
    return config

class CustomizedGaussianNB(GaussianNB):
    """Cast dtype to 128 float to avoid numerical underflow"""
    def __init__(self):
        super().__init__()

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test vector X.
        Overriding function. Check value in jll and set small value to -inf.
        Cause np.exp(very small value) will cause numerical underflow.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        jll = self._joint_log_likelihood(X)
        # normalize by P(x) = P(f_1, ..., f_n)

        # Replace all value smaller than -10000 to -np.inf, when the shape have
        # more than 1 column
        if len(jll.shape) > 1 and jll.shape[1] > 1:
            jll[jll<=-10000] = -np.inf
        jll = jll.astype('float128')
        log_prob_x = logsumexp(jll, axis=1)
        jll = jll.astype('float128')
        log_prob_x = log_prob_x.astype('float128')
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """

        return np.exp(self.predict_log_proba(X).astype('float128'))
