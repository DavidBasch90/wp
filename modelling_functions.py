from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import numpy as np
import pandas as pd

np.random.seed(123)



def run_bootstrapped_rf_model(data, cl_method):
    #prepare the data
    model_df = data[data['method']==cl_method]
    model_df['first_cluster'] = model_df['first_cluster'].astype('category')
    model_df = model_df.drop(columns=['second_cluster', 'method'])

    #bootstrap resampling
    bootstrap_sample = resample(model_df)

    #fit random forest model
    rf = RandomForestClassifier(n_estimators=1000)
    X = bootstrap_sample.drop('first_cluster', axis=1)
    y = bootstrap_sample['first_cluster']
    rf.fit(X,y)

    #return the model_df

    return rf


def get_importance_scores(data, cl_method):
    #prepare
    model_df = data[data['method']==cl_method]
    model_df['first_cluster']=model_df['first_cluster'].astype('category')
    model_df = model_df.drop(columns=['second_cluster', 'method'])

    #fit random forest model

    rf = RandomForestClassifier(n_estimators=1000)
    X = model_df.drop('first_cluster',axis=1)
    y = model_df['first_cluster']
    rf.fit(X,y)

    #get feature importances

    importances = rf.feature_importances_
    return importances

def run_subcluster_bootstrapped_rf_model(data,cl_method,cluster_num):
    #prepare
    model_df=data[(data['method']==cl_method)&(data['first_cluster']==cluster_num)]
    model_df['second_cluster'] = model_df['second_cluster'].astype('category')
    model_df = model_df.drop(columns=['first_cluster', 'method'])

    #create bootstrap samples
    bootstrap_sample = resample(model_df)

    #fit random forest model
    rf = RandomForestClassifier(n_estimators=1000)
    X = bootstrap_sample.drop('second_cluster',axis=1)
    y = bootstrap_sample['second_cluster']

    #return the fitted model
    rf.fit(X,y)

def get_subcluster_importance_scores(data, cl_method, cluster_num):
    #prep data
    model_df = data[(data['method']==cl_method)&(data['first_cluster']==cluster_num)]
    model_df['second_cluster'] = model_df['second_cluster'].astype('category')
    model_df = model_df.drop(columns=['first_cluster','method'])

    #fit random forest model
    rf = RandomForestClassifier(n_estimators=1000)
    X = model_df.drop('second_cluster', axis=1)
    y = model_df['second_cluster']
    rf.fit(X,y)
