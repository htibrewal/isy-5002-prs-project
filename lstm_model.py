import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense

# import tensorflow as tf

from prepare_data_lstm import prepare_resultant_df_v3, create_sequences

output_scaler = MinMaxScaler()

resultant_df = prepare_resultant_df_v3(output_scaler)

# Look-back window of 12 time steps
n_steps = 12
X_all, y_all = [], []

# Process each parking lot separately
for lot in resultant_df['car_park_number'].unique():
    lot_data = resultant_df[resultant_df['car_park_number'] == lot].drop(columns=['car_park_number']).values
    X, y = create_sequences(lot_data, n_steps)
    X_all.append(X)
    y_all.append(y)

# Combine data for all parking lots
X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=42)

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))  # Predicting available spaces

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(tf.keras.layers.Dense(16, activation='relu'))
# model.add(tf.keras.layers.Dense(1))  # Predicting available spaces

model.compile(optimizer='adam', loss='mse')

print(model.summary())

# Step 7: Train the model
history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, verbose=1)

# Step 8: Evaluate the model
y_pred = model.predict(X_test)

y_test = y_test.reshape(-1, 1)
y_pred = y_pred.reshape(-1, 1)

y_test_actual = output_scaler.inverse_transform(y_test)
y_pred_actual = output_scaler.inverse_transform(y_pred)

mse = mean_squared_error(y_test_actual, y_pred_actual)
print(f'Mean Squared Error (MSE): {mse}')

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test_actual, y_pred_actual)
print(f'Mean Absolute Error (MAE): {mae}')


errors = y_test_actual - y_pred_actual

# plotting the error distribution
plt.figure(figsize=(10, 6))
sns.histplot(errors, bins=100, kde=True)
plt.title('Error Distribution')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
# plt.show()

plt.savefig('error_distribution.png', dpi=500)
plt.close()


# # plotting heatmap of prediction errors (aggregated by lat and long)
# df_eval = pd.DataFrame({'latitude': X_test['x_coord'], 'longitude': X_test['y_coord'], 'error': errors})
#
# # Create a pivot table to aggregate the error by latitude and longitude
# heatmap_data = df_eval.pivot_table(index='latitude', columns='longitude', values='error', aggfunc=np.mean)
#
# plt.figure(figsize=(10, 8))
# sns.heatmap(heatmap_data, cmap='coolwarm', center=0)
# plt.title('Heatmap of Prediction Error by Location')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# # plt.show()
#
# plt.savefig('error_heatmap.png', dpi=500)
# plt.close()
