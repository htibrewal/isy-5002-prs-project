from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score

from prepare_data_v2 import prepare_resultant_df_v2

resultant_df = prepare_resultant_df_v2()
# resultant_df = resultant_df[:10000]

X = resultant_df.drop(columns=['car_park_number'])
# y = resultant_df['car_park_number']

Z = linkage(X, method='ward')

best_k = 0
best_silhouette_score = 0

for k in range(2, 20):
    cluster_labels = fcluster(Z, k, criterion='maxclust')

    # Calculate the Silhouette Score
    silhouette_avg = silhouette_score(X, cluster_labels)

    if silhouette_avg > best_silhouette_score:
        best_silhouette_score = silhouette_avg
        best_k = k


print(f"Best Silhouette score is {best_silhouette_score: .2f}")
print(f"Best no of clusters is {best_k}")


plt.figure(figsize=(10, 5))
dendrogram(Z, labels=resultant_df['car_park_number'].values)
plt.title('Hierarchical Clustering Dendrogram (Available Lots > 0)')
plt.xlabel('Parking Lot')
plt.ylabel('Distance')
plt.savefig('hierarchical_clustering.png')
