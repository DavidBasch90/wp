import pandas as pd
import numpy as np
from sklearn.decomposition import PCA 
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

def clean_wimd_data(df):
    na_codes = ["BURG","CRDG","FIRE", "THEF"]
    df = df.replace(-9999.0, np.nan)
    #drop aggregate rows?
    df = df[df.LSOA_Code != 'Wales']
    #pivot table
    df=df.pivot(index='LSOA_Code', columns='Indicator_Code',values='Data')
    #drop na_codes
    df = df.drop(na_codes,axis=1)


    return df

def impute_missing_wimd_values(df):
    np.random.seed(234)

    #impute missing values 
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns = df.columns, index = df.index)
    return df_imputed
    
def manual_dim_reduction(df):
    
    cols_to_drop = ['AIQP1','LLTI','HQUA']
    wimd_25 = df.drop(columns=cols_to_drop) 
    return wimd_25

def run_travel_pca(df):
    travel_codes = ['PRFS','PRGP','PRLI','PRPE','PRPH','PRPO','PRPS','PRSF','PRSS','PUFS','PUGP','PULI','PUPH','PUPO','PUPS','PUSF','PUSS']
    travel_df = df[travel_codes]
    #normalise
    scaler = StandardScaler()
    travel_df_scaled = scaler.fit_transform(travel_df)

    #pca
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(travel_df_scaled)

    #add pca results to df
    df['PCA_Travel'] = pca_result
    df = df.drop(travel_codes,axis=1)
    return df
    


def scale(x,na_rm=True):
    return zscore(x,nan_policy='omit' if na_rm else 'propagate')