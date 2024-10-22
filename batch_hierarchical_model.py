import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from prepare_data_v2 import prepare_resultant_df_v2
from setup import get_X_y_encoded


# Mini-batch clustering
def mini_batch_hierarchical(X, batch_size=10000, n_clusters=10):
    # Process data in batches
    labels = np.zeros(len(X))
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        if len(batch) < 2:  # Skip batches that are too small
            continue

        # Perform clustering on the batch
        clustering = AgglomerativeClustering(n_clusters=min(n_clusters, len(batch)))
        batch_labels = clustering.fit_predict(batch)

        # Assign labels to the original dataset
        labels[i:i + len(batch)] = batch_labels

    return labels


def plot_silhouette(n_clusters_range, silhouette_scores):

    plt.figure(figsize=(10, 6))
    plt.plot(n_clusters_range, silhouette_scores, 'bo-')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.savefig('silhouette_score.png')


def main():
    resultant_df = prepare_resultant_df_v2()
    resultant_df = resultant_df[:100000]

    X, _ = get_X_y_encoded(resultant_df)

    best_n_clusters = 0
    best_silhouette_score = 0
    silhouette_scores = []

    for n_clusters in range(2, 21):
        y_pred = mini_batch_hierarchical(X, n_clusters=n_clusters)

        silhouette_avg = silhouette_score(X, y_pred)
        print("n_clusters =", n_clusters, "Average silhouette_score =", silhouette_avg)

        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_n_clusters = n_clusters

        silhouette_scores.append(silhouette_avg)

    print(f"Best Silhouette score is {best_silhouette_score: .2f}")
    print(f"Best no of clusters is {best_n_clusters}")

    plot_silhouette(range(2, 21), silhouette_scores)

if __name__ == '__main__':
    main()