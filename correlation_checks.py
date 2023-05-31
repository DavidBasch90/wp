###R file -> correlation_checks.R

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


def cor_checks():
    #load data
    filename = 'test.csv'
    wimd_2019 = pd.read_csv(filename)

    ######################
    #data processing

    #PCA analysis
    pca_cols = ['col1','col2']
    pca = PCA()
    wimd_2019[pca_cols] = pca.fit_transform(wimd_2019[pca_cols])

    #re order the columns
    cols = list(wimd_2019)
    cols.insert(0,cols.pop(cols.index('LSOA_Code')))
    wimd_2019 = wimd_2019.loc[:,cols]

    #scale
    scaler = StandardScaler()
    scale_columns = ['DIG','PCA_Travel']
    wimd_2019[scale_columns] = scaler.fit_transform(wimd_2019[scale_columns])

    #dropunused columns
    wimd_processed = wimd_2019.drop(columns=['LLTI','AIQP2'])

    #######################
    #Correlation analysis
    correlation = wimd_processed.drop(columns=['LSOA_Code']).corr()

    #flatten the matrix
    correlation_long = correlation.unstack().reset.index()
    correlation_long.columns = ['term','name','value']

    #create pair column 
    correlation_long['pair'] = correlation_long['term']+"-"+correlation_long['name']
    correlation_long = correlation_long[~correlation_long['value'].isna()]

    #get top 10 correlations

    top_10_correlations = correlation_long.nlargest(10,'value',keep='all')


    #data vis

    plt.figure(figsize=(10,8))
    sns.barplot(data=top_10_correlations,x='value',y='pair',hue='term',dodge=False)
    plt.show()