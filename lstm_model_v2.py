import numpy as np
import seaborn as sns
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from create_optimised_lstm import create_optimised_lstm
from prepare_data_lstm import prepare_resultant_df_v3
from sequential_data_gen import create_train_val_generators

# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

output_scaler = MinMaxScaler()
resultant_df = prepare_resultant_df_v3(output_scaler, use_static_features=False)

# batch_size = 256
batch_size = 64
n_steps = 10
train_gen, val_gen, test_gen = create_train_val_generators(resultant_df, n_steps=n_steps, batch_size=batch_size)

sample_batch_X, sample_batch_y = train_gen[0]
print(f"Sample batch shapes:")
print(f"X shape: {sample_batch_X.shape}")  # Should be (batch_size, n_steps, n_features)
print(f"y shape: {sample_batch_y.shape}")  # Should be (batch_size,)
print(sample_batch_y[0])

n_features = len(resultant_df.columns) - 1
input_shape = (n_steps, n_features)
model = create_optimised_lstm(input_shape)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    # use_multiprocessing=True,
    # workers=4,
    callbacks=[
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]
)

y_pred = model.predict(test_gen)

y_test = np.array([])
for i in range(len(test_gen)):
    _, batch_y = test_gen[i]
    y_test = np.concatenate([y_test, batch_y])

# Reshape for inverse transform
y_test = y_test.reshape(-1, 1)
y_pred = y_pred.reshape(-1, 1)

# Inverse transform if you used scaling
y_test_actual = output_scaler.inverse_transform(y_test)
y_pred_actual = output_scaler.inverse_transform(y_pred)

# Calculate metrics
mse = mean_squared_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_pred_actual)

# Print metrics
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')

r2 = r2_score(y_test_actual, y_pred_actual)
print(f"R-squared Score: {r2:.4f}")


plt.figure(figsize=(12, 6))
plt.plot(y_test_actual[:100], label='Actual')
plt.plot(y_pred_actual[:100], label='Predicted')
plt.title('Actual vs Predicted Values (First 100 samples)')
plt.legend()
plt.grid(True)
plt.savefig('timeseries.png', dpi=500)


errors = y_test_actual - y_pred_actual

# plotting the error distribution
plt.figure(figsize=(10, 6))
sns.histplot(errors, bins=50, kde=True)
plt.title('Prediction Error Distribution')
plt.xlabel('Prediction Error')

plt.savefig('error_distribution.png', dpi=500)
plt.close()
