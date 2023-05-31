#pam clustering script

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform



def load_and_process_data(file_path, index_col, drop_cols):
    '''
    load the data from csv, set the index, 
    drop the desired columsn, scale the numeric features w/ standardscaler
    
    '''
    #read and prep input
    data = pd.read_csv(file_path)
    data.set_index(index_col, inplae=True)
    data.drop(columns=drop_cols, inplace=True)

    #standardise
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)

    return data


def fit_kmeans(data, n_clusters):
    '''
    Takes two inputs: a dataframe and the number of clusters you want, then applies k means clustering,
    Returns DataFrame with additional cluster assignment column
    
    '''
    kmeans = KMeans(n_clusters=n_clusters, random_state=123)
    labels = kmeans.fit_predict(data)
    labelled_data = data.copy()
    labelled_data['cluster'] = labels

    return labelled_data

def get_subcluster_assignments(data, labels, cluster_num, k_num):
    '''
    break the main clusters into sub clusters, 
    Returns df of subcluster labels. Will be called in the next function
    
    '''

    #select data in the given cluster
    data_sub = data[labels['cluster']==cluster_num]

    #fit k-means on the subset
    kmeans_sub = KMeans(n_clusters=k_num, random_state = 123)
    labels_sub = kmeans_sub.fit_predict(data_sub)
    labels_sub = pd.DataFrame(labels_sub,index=data_sub.index,columns=['subcluster'])

    return labels_sub

def process_clusters(clustered_data, n_subclusters):
    '''
    This will be where the sub clusters are merged into one dataframe. it creates sub clusters within each 
    main cluster and returns a dataframe with both main and sub cluster assignment
    
    '''
    #get the main labels
    labels = clustered_data[['cluster']]

    #init empty df to hold the subcluster labels
    subcluster_labels = pd.DataFrame()

    #get subcluster assignments for each main cluster
    for cluster_num in labels['cluster'].unique():
        subclusters = get_subcluster_assignments(clustered_data.drop(columns='cluster'),labels,cluster_num,n_subclusters)
        subcluster_labels = pd.concat([subcluster_labels, subclusters])

    #join main cluster and subcluster labels
    labels = labels.join(subcluster_labels)

    return labels

def process_data_and_save_models(file_path,index_col,drop_cols,n_clusters,n_subclusters):
    '''
    Main driver function - will run all the above functions (loading data, fitting the kmeans, processing clusters and saving it all)
    '''
    #load and process data
    data = load_and_process_data(file_path,index_col,drop_cols)

    #fit kmeans and get cluster labels
    clustered_data = fit_kmeans(data, n_clusters)

    #process subclusters and get the final labels
    final_labels = process_clusters(clustered_data, n_subclusters)

    #save the final labels and the processed data 
    final_labels.to_csv('final_labels.csv')
    clustered_data.to_csv('clustered_data.csv')

    return
    
