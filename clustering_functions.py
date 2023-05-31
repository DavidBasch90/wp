import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from fancyimpute import IterativeImputer
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from scipy.spatial import distance
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import pdist, squareform

def get_best_model(df):
    weights = {
        'silhouette_avg':1,
        'avg_connectivity':1,
        'dunn_index':1
    }
    #apply weights
    for measure, weight in weights.items():
        df[measure] *= weight
    #calculate the weighted score
    max_distance = df["avg_connectivity"].max()
    df['score'] =df['silhouette_avg']+df['dunn_index']-df["avg_connectivity"]
    best_model = df.loc[df['score'].idxmax()]

    df.to_csv('./output/clustering_tests/clustering_results.csv', index=False)

    return best_model


def clustering_tests(wimd_25,wimd_28,wimd_pc_8,wimd_pc_12):

    datasets = {
        "wimd_25": wimd_25,
        "wimd_28": wimd_28,
        "wimd_pc_8": wimd_pc_8,
        "wimd_pc_12": wimd_pc_12
    }

    results = []

    for name, data in datasets.items():
        for n_clusters in range(2, 15):
            for model_name, model in [
                ("kmeans", KMeans(n_clusters=n_clusters, random_state=123)),
                ("agnes", AgglomerativeClustering(n_clusters=n_clusters)),
                ("kmeds", KMedoids(n_clusters=n_clusters, random_state=123))
            ]:
                labels = model.fit_predict(data)
                silhouette_avg = silhouette_score(data, labels)
                ##n_neighbors are not clusters - 
                connectivity = kneighbors_graph(data, n_neighbors=n_clusters).toarray()
                
                #compute connectivity
                avg_intra_cluster_distance = []
                for cluster in np.unique(labels):
                    cluster_points = connectivity[labels==cluster]
                    if len(cluster_points)> 1:
                        avg_intra_cluster_distance.append(np.mean(distance.pdist(cluster_points)))

                # Corrected Dunn index calculation
                inter_cluster_distances = []
                intra_cluster_distances = []
                for cluster in np.unique(labels):
                    intra_cluster_data = data[labels == cluster]
                    print(f"cluster {cluster} size: {len(intra_cluster_data)}")
                    if len(intra_cluster_data) > 1:
                        intra_cluster_distances.append(pdist(intra_cluster_data).max())
                    for other_cluster in np.unique(labels):
                        if cluster != other_cluster:
                            inter_cluster_data = data[labels == other_cluster]
                            if len(inter_cluster_data)>0 and len(intra_cluster_data)>0:
                                inter_cluster_distances.append(distance.cdist(intra_cluster_data, inter_cluster_data).min())
                if intra_cluster_distances and inter_cluster_distances:

                    dunn_index = min(inter_cluster_distances) / max(intra_cluster_distances)
                else:
                    dunn_index = np.nan
                
                results.append({
                    "dataset": name,
                    "clusters": n_clusters,
                    "model": model_name,
                    "silhouette_avg": silhouette_avg,
                    "avg_connectivity": np.mean(avg_intra_cluster_distance) if avg_intra_cluster_distance else np.nan,
                    "dunn_index": dunn_index
                })

    results_df = pd.DataFrame(results)

    # Plotting
    for measure in ["silhouette_avg", "avg_connectivity", "dunn_index"]:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=results_df, x="clusters", y=measure, hue="dataset", style="model")
        plt.title(measure)
        plt.savefig(f"./output/clustering_tests/{measure}.png")
        plt.close()
    
    best_model = get_best_model(results_df)
    print(best_model)


def run_travel_pcab(df, keep_travel_vars=False):
    '''
    performs data cleaning, applies PCA on travel vars

    '''

    #get indicators and clean the columns

    generic_patten = ["return "," time to a", '''\\(minutes)\\)''']

    df = df.rename(columns={'Indicator_ItemName_ENG':'indicator_name'})
    df = df.columns.str.lower()

    for pat in generic_patten:
        df['indicator_name'] =df['indicator_name'].str.replace(pattern,"")
    
    indicator_lookup = df['indicator_code', 'indicator_name'].drop_duplicates()

    #reshape the dataframe

    df['Data'] = df['Data'].replace(-9999.0, np.nan)
    df_wide = df.pivot(index='LSOA_Code', columns='Indicator_Code', values='Data')
    na_codes = ["OBCH","BURG","CRDG", "FIRE", "THIEF"]
    df_wide = df_wide.drop(na_codes,axis=1)

    #run the PCA

    travel_codes = indicator_lookup[indicator_lookup['indicator_name'].str.contains('travel')]['indicator_code']
    travel_variables = df_wid[travel_codes]

    #standardise the data
    scaler = StandardScaler()
    travel_variables_std = scaler.fit_transform(travel_variables)

    #run PCA
    pca_output = run_pca(travel_variables_std)

    #merge resuts back into the df
    df__wide['PCA_Travel'] = pca_output

    #do we need to reemove the travel vars?
    '''
    if not keep_travel_vars:
        df_wide = df_wide.drop(travel_codes,axis=1)
    '''
    return df_wide


def run_full_pca(df):
    '''
    performs data cleaning, applies PCA to whole dataset

    '''
    

    #select the numeric data only for pca
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    #scale the data for PCA
    scaler = StandardScaler()
    df_std = scaler.fit_transform(df[numeric_cols])

    #run pc on scaled data
    pca = PCA()
    pca_output = pca.fit_transform(df_std)

    #plot the pca
    explained_variance = pca.explained_variance_ratio_
    plt.figure(figsize=(8,6))
    plt.plot(range(1,len(explained_variance)+1),explained_variance, marker='o', linestyle='-')
    plt.title('Scree plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.savefig('./output/PCA/full/scree-plot-proportion-variance-explained.png')

    #cumulative explained variance plot
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8,6))
    plt.plot(range(1,len(cumulative_explained_variance)+1),cumulative_explained_variance, marker='o', linestyle='-')
    plt.title('Cumulative Explained Variance Plot')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Proportion of Variance Explained')
    plt.savefig('./output/PCA/full/cumsum-proportion-of-variance-explained.png')
    #convert the pca output into a dataframe

    pca_df = pd.DataFrame(data=pca_output, columns=['PC'+str(i) for i in range(1,pca_output.shape[1]+1)], index=df.index)

    df = pd.concat([df,pca_df],axis=1)
    return df, pca


def select_top_n_components(df,pca,n_components):
    top_n_components = ['PC'+str(i) for i in range(1,n_components+1)]
    df_top_n = df[top_n_components]
    return df_top_n

######################
#above could be made into one function

def scale2(x, na_rm=True):
    '''
    standardises the series (- mean and / std)

    '''
    if na_rm:
        x=x[~np.isnan(x)]
    return (x - np.mean(x))/np.std(x)

def plot_stability_measures(stability_object):
    '''
    Plot the stability measures -- APN, AD, ADM, FOM

    '''
    pass



