from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

from prepare_data_v2 import prepare_resultant_df_v2
from setup import get_train_test_X_y

resultant_data = prepare_resultant_df_v2()
resultant_data = resultant_data[:1000000]

X_train, X_test, y_train, y_test = get_train_test_X_y(resultant_data, test_size=0.3)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_test, y_pred))
