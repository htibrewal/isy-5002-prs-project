import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from prepare_data import prepare_resultant_df
from setup import get_X_y_encoded


resultant_df = prepare_resultant_df()
X, y_encoded = get_X_y_encoded(resultant_df)

# train and test split for X & y
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

best_accuracy = 0
best_model = None

for n_estimators in [50, 100, 200, 500]:
    print("No of estimators = ", n_estimators)
    random_forest_model = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=5)
    random_forest_model.fit(X_train, y_train)

    y_pred = random_forest_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy = ", accuracy)

    # prints classification matrix for each class
    # print(classification_report(y_test, y_pred))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = random_forest_model


print()
print("Best accuracy = ", best_accuracy)
print("Feature importance of best model = ", pd.Series(best_model.feature_importances_, index=X.columns))
