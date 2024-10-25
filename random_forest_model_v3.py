import gc

import dask.dataframe as dd
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import parallel_backend, dump
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from prepare_data_v2 import prepare_resultant_df_v2
from setup import get_scaler_and_encoder


def split_dataset(dataframe, n_chunks=15, test_size=0.2):
    dataframe = optimise_dataframe(dataframe)

    train_df, test_df = train_test_split(dataframe, test_size=test_size, random_state=42, shuffle=True)

    # total_rows = len(dataframe)
    # ddf = dd.from_pandas(dataframe, chunksize=total_rows//n_chunks)
    #
    # test_rows = int(total_rows * test_size)
    #
    # ddf['_row_no'] = 1
    # ddf['_row_no'] = ddf._row_no.cumsum()

    # test_ddf = ddf[ddf._row_no <= test_rows]
    # train_ddf = ddf[ddf._row_no > test_rows]
    #
    # train_ddf = train_ddf.drop('_row_no', axis=1)
    # test_ddf = test_ddf.drop('_row_no', axis=1)

    train_ddf = dd.from_pandas(train_df, chunksize=len(train_df)//n_chunks)
    test_ddf = dd.from_pandas(test_df, chunksize=len(test_df)//n_chunks)

    print(f"Number of train partitions: {train_ddf.npartitions}")
    print(f"Number of test partitions: {test_ddf.npartitions}")

    train_ddf.to_parquet('train.parquet')
    test_ddf.to_parquet('test.parquet')

    del dataframe, train_ddf, test_ddf
    gc.collect()

    return 'train.parquet', 'test.parquet'

def optimise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    optimised_df = df.copy()

    for col in df.columns:
        col_type = optimised_df[col].dtype

        if col_type != 'object':
            c_min = optimised_df[col].min()
            c_max = optimised_df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    optimised_df[col] = optimised_df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    optimised_df[col] = optimised_df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    optimised_df[col] = optimised_df[col].astype(np.int32)

            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    optimised_df[col] = optimised_df[col].astype(np.float32)
                else:
                    optimised_df[col] = optimised_df[col].astype(np.float64)

    final_memory = optimised_df.memory_usage(deep=True).sum() / 1024 ** 2
    memory_reduction = initial_memory - final_memory

    print(f"Memory reduced by: {memory_reduction:.2f} MB ({(memory_reduction / initial_memory) * 100: .2f}%)")

    return optimised_df


def train_random_forest(train_data_path, target_column, label_encoder, n_estimators = 50):
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=5,
        n_jobs=-1,
        verbose=1,
        warm_start=True,
        random_state=42
    )

    train_ddf = dd.read_parquet(train_data_path)
    total_chunks = train_ddf.npartitions

    for i, chunk in enumerate(train_ddf.partitions):
        chunk = chunk.compute()

        if len(chunk) == 0:
            print(f"Warning: Empty chunk encountered in partition {i + 1}")
            continue

        X_chunk, y_chunk = chunk.drop(columns=target_column), label_encoder.fit_transform(chunk[target_column])
        print(f"X_chunk shape: {X_chunk.shape}")
        print(f"y_chunk shape: {y_chunk.shape}")

        print(f"Training on chunk {i + 1} of {total_chunks}")
        rf_model.fit(X_chunk, y_chunk)

        del X_chunk, y_chunk, chunk
        gc.collect()

    return rf_model


def predict(model, test_data_path, target_column, label_encoder):
    predicted_values = []
    actual_values = []
    predicted_probs = []

    test_ddf = dd.read_parquet(test_data_path)
    total_chunks = test_ddf.npartitions

    for i, chunk in enumerate(test_ddf.partitions):
        chunk = chunk.compute()

        if len(chunk) == 0:
            print(f"Warning: Empty chunk encountered in partition {i + 1}")
            continue

        X_test = chunk.drop(columns=target_column)
        print(f"X shape: {X_test.shape}")
        chunk_predictions = model.predict(X_test)
        print(f"y_predictions shape: {chunk_predictions.shape}")

        predicted_values.extend(chunk_predictions)
        actual_values.extend(label_encoder.fit_transform(chunk[target_column]))

        # extend list of prediction probabilities
        predicted_probs.extend(model.predict_proba(X_test))

        print(f"Processing chunk {i + 1} of {total_chunks}")

        del chunk, X_test
        gc.collect()

    # np.save('predicted_probs.npy', predicted_probs)

    evaluate(actual_values, predicted_values)

    return predicted_values


def evaluate(actual_values, predicted_values):
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)

    # Calculate and print classification metrics
    print("\nClassification Report:")
    print(classification_report(actual_values, predicted_values))

    print("\nAccuracy Score:", accuracy_score(actual_values, predicted_values))

    # Plot confusion matrix
    classes = np.unique(actual_values)
    plot_confusion_matrix(actual_values, predicted_values, labels=classes)


def plot_confusion_matrix(actual_values, predicted_values, labels=None):
    matrix = confusion_matrix(actual_values, predicted_values, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels if labels is not None else 'auto',
        yticklabels=labels if labels is not None else 'auto'
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()


if __name__ == '__main__':
    scaler, categorical_encoder, label_encoder = get_scaler_and_encoder()
    resultant_data = prepare_resultant_df_v2(scaler, categorical_encoder, use_mean_sampling=True)
    # resultant_data = resultant_data[:100000]

    no_chunks = 5
    train_path, test_path = split_dataset(resultant_data, n_chunks=no_chunks)

    target_column = 'car_park_number'
    # feature_columns = resultant_data.columns.drop(labels=[target_column])

    with parallel_backend('threading', n_jobs=-1):
        model = train_random_forest(train_path, target_column, label_encoder)

    model_data = {
        'model': model,
        'scaler': scaler,
        'categorical_encoder': categorical_encoder,
        'label_encoder': label_encoder
    }

    dump(model_data, 'random_forest_model.pkl')

    predictions, prediction_probs = predict(model, test_path, target_column, label_encoder)
