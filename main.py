from clustering_functions import *
from data_preprocessing import *
from modelling_functions import *
from correlation_checks import *
from pam_cluster_assignments import *

import pandas as pd



def main():

    '''
    Cleaning and data preprocessing
    '''
    #read data
    filename = 'wimd_2019.csv'
    data = pd.read_csv(f'./input/{filename}')
    #preprocessing
    data = clean_wimd_data(data)
    
    #imputation
    data = impute_missing_wimd_values(data)

    '''
    Dataframe generation and preparation for clustering
    '''

    #run PCA on full dataset
    data_pca, pca = run_full_pca(data)
    #create 8 component model
    wimd_8 = select_top_n_components(data_pca,pca, 8)
    #create 12 component model
    wimd_12 = select_top_n_components(data_pca,pca, 12) 
    #28 col model after travel pca applied
    wimd_28 = run_travel_pca(data)
    #25 col model after WIMD_advised cols removed
    wimd_25 = manual_dim_reduction(wimd_28)

    '''
    Internal validation measures
    '''

    clustering_tests(wimd_25,wimd_28,wimd_8,wimd_12)
    





if __name__ == "__main__":
    main()