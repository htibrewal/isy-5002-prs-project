from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer

from prepare_data import prepare_resultant_df
from setup import get_X_y_encoded


resultant_df = prepare_resultant_df()
resultant_df = resultant_df[:10000]
X, y_encoded = get_X_y_encoded(resultant_df)

k_means = None
for k in range(2, 19):
    k_means = KMeans(n_clusters=k, n_init=5, max_iter=100, random_state=42)
    y_pred = k_means.fit_predict(X)

    silhouette_avg = silhouette_score(X, y_pred)
    print("n_clusters =", k, "Average silhouette_score =", silhouette_avg)


# initialise KElbowVisualizer and plot distortion score elbow
cluster = KMeans(n_init=5, max_iter=100, random_state=42)
visualizer = KElbowVisualizer(cluster, k=(2, 19))

visualizer.fit(X)    # Fit the data to the visualizer
visualizer.show()

