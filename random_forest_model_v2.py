import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from prepare_data_v2 import prepare_resultant_df_v2
from setup import get_train_test_X_y, get_scaler_and_encoder

scaler, categorical_encoder, label_encoder = get_scaler_and_encoder()
resultant_data = prepare_resultant_df_v2(scaler, categorical_encoder, use_mean_sampling=True)
# resultant_data = resultant_data[:10000]

X_train, X_test, y_train, y_test = get_train_test_X_y(resultant_data, label_encoder, test_size=0.3)

best_accuracy = 0
best_model = None

for n_estimators in [50, 100, 200]:
    print("No of estimators = ", n_estimators)
    random_forest_model = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=5)
    random_forest_model.fit(X_train, y_train)

    y_pred = random_forest_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy = ", accuracy)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = random_forest_model

print("\nBest accuracy = ", best_accuracy)

columns = resultant_data.columns.drop(labels=['car_park_number'])
print("Feature importance of best model\n", pd.Series(best_model.feature_importances_, index=columns))

model_data = {
    'model': best_model,
    'scaler': scaler,
    'categorical_encoder': categorical_encoder,
    'label_encoder': label_encoder
}

with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


