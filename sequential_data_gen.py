import numpy as np
from keras.api.utils import Sequence


class SequentialDataGenerator(Sequence):
    def __init__(self, df, lot_numbers, n_steps, batch_size=32, shuffle=True):
        self.df = df
        self.lot_numbers = lot_numbers
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.sequence_indices = self._calculate_sequence_indices()

        if self.shuffle:
            np.random.shuffle(self.sequence_indices)

    def _calculate_sequence_indices(self):
        indices = []

        for lot in self.lot_numbers:
            lot_data = self.df[self.df['car_park_number'] == lot]
            lot_len = len(lot_data)

            if lot_len > self.n_steps:
                indices.extend([
                    (lot, i) for i in range(lot_len - self.n_steps)
                ])

        return indices

    def __len__(self):
        return int(np.ceil(len(self.sequence_indices) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.sequence_indices[index * self.batch_size:(index + 1) * self.batch_size]

        X_batch = []
        y_batch = []

        for lot, start_idx in batch_indices:
            lot_data = self.df[self.df['car_park_number'] == lot].drop(columns='car_park_number').values

            X = lot_data[start_idx:start_idx + self.n_steps]
            y = lot_data[start_idx + self.n_steps, 0]

            X_batch.append(X)
            y_batch.append(y)

        return np.array(X_batch), np.array(y_batch)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.sequence_indices)


def create_train_val_generators(df, n_steps=12, batch_size=32, val_split=0.3):
    car_park_numbers = df['car_park_number'].unique()
    n_lots = len(car_park_numbers)
    n_train = int(n_lots * (1 - val_split))

    np.random.shuffle(car_park_numbers)
    train_lots = car_park_numbers[:n_train]
    val_lots = car_park_numbers[n_train:]

    train_generator = SequentialDataGenerator(df, train_lots, n_steps, batch_size=batch_size)
    val_generator = SequentialDataGenerator(df, val_lots, n_steps, batch_size=batch_size)

    # using validation lots for testing as well
    test_generator = SequentialDataGenerator(df, val_lots, n_steps, batch_size=batch_size, shuffle=False)

    return train_generator, val_generator, test_generator
