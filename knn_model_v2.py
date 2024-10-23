import pickle

from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

from prepare_data_v2 import prepare_resultant_df_v2
from setup import get_train_test_X_y, get_scaler_and_encoder

scaler, categorical_encoder, label_encoder = get_scaler_and_encoder()

resultant_data = prepare_resultant_df_v2(scaler, categorical_encoder, use_mean_sampling=True)
resultant_data = resultant_data[:10000]

X_train, X_test, y_train, y_test = get_train_test_X_y(resultant_data, label_encoder, test_size=0.3)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

model_data = {
    'model': knn_model,
    'scaler': scaler,
    'categorical_encoder': categorical_encoder,
    'label_encoder': label_encoder
}

with open('knn_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_test, y_pred))
