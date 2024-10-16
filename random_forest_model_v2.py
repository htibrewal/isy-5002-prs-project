import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from prepare_data_v2 import prepare_resultant_df_v2
from setup import get_train_test_X_y

resultant_df = prepare_resultant_df_v2()

X_train, X_test, y_train, y_test = get_train_test_X_y(resultant_df, test_size=0.3)

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


print()
print("Best accuracy = ", best_accuracy)
print("Feature importance of best model = ", pd.Series(best_model.feature_importances_, index=X_train.columns))
