from keras.api.models import Sequential
from keras.api.layers import LSTM, BatchNormalization, Dropout, Dense
from keras.api.optimizers import Adam

# import tensorflow as tf
# from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout
# from tensorflow.keras.optimizers import Adam


def create_optimised_lstm(input_shape, learning_rate=0.01):
    # model = tf.keras.Sequential([
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(16, activation='relu'),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    print(model.summary())

    return model
