from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from prepare_data import prepare_resultant_df
from setup import get_train_test_X_y

resultant_df = prepare_resultant_df()

X_train, X_test, y_train, y_test = get_train_test_X_y(resultant_df, test_size=0.3)

# create XGBoost Classifier model and fit on the training data
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
