from sklearn.cluster import KMeans

def perform_clustering(X, method: str):

    model = None
    labels = None

    if method == 'kmeans':
        k = 3
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)
    else:
        raise ValueError(f"Clustering method '{method}' is not supported.")

    
    return model, labels
        