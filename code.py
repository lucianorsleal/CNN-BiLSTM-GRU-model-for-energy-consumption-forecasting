from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import tensorflow as tf
import os

region = 'ne'
file_path = '/content/drive/MyDrive/timeseries/splitted_dataset_region_ne.csv'
model_dir = '/content/drive/MyDrive/timeseries/trainnedmodels'
os.makedirs(model_dir, exist_ok=True)

all_predictions = {}
all_test_normalized = {}

def process_region(region, file_path):
    cnn_lstm_params = (64, 3, 50, 'tanh', 'adam', 64)
    data = pd.read_csv(file_path, parse_dates=['din_instante'], index_col='din_instante')
    series = data['val_cargaenergiahomwmed'].values

    scalers = {'minmax': MinMaxScaler(), 'standard': StandardScaler(), 'robust': RobustScaler()}
    scaler = scalers['minmax']
    series_normalized = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    train_size = int(len(series_normalized) * 0.8)
    train_normalized = series_normalized[:train_size]
    test_normalized = series_normalized[train_size:]

    window_size = 30
    X_train, y_train = [], []
    for i in range(len(train_normalized) - window_size):
        X_train.append(train_normalized[i:i + window_size])
        y_train.append(train_normalized[i + window_size])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model_list = []

    def build_and_train(model_name, model_layers):
        model = tf.keras.Sequential(model_layers)
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)
        model.save(os.path.join(model_dir, f"{model_name}.keras"))
        model_list.append((model_name, model))

    # Architectural Ablation Models
    build_and_train('CNN', [
        tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(window_size, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])

    build_and_train('BiLSTM', [
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=False), input_shape=(window_size, 1)),
        tf.keras.layers.Dense(1)
    ])

    build_and_train('GRU', [
        tf.keras.layers.GRU(50, input_shape=(window_size, 1)),
        tf.keras.layers.Dense(1)
    ])

    build_and_train('CNN_BiLSTM', [
        tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(window_size, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=False)),
        tf.keras.layers.Dense(1)
    ])

    build_and_train('CNN_GRU', [
        tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(window_size, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.GRU(50),
        tf.keras.layers.Dense(1)
    ])

    build_and_train('BiLSTM_GRU', [
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True), input_shape=(window_size, 1)),
        tf.keras.layers.GRU(50),
        tf.keras.layers.Dense(1)
    ])

    build_and_train('CNN_BiLSTM_GRU', [
        tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(window_size, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True)),
        tf.keras.layers.GRU(50),
        tf.keras.layers.Dense(1)
    ])

    # Layer Depth Ablation Example: CNN (3 layers)
    build_and_train('CNN_2Layers', [
        tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(window_size, 1)),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])


    # Layer Depth Ablation Example: CNN (3 layers)
    build_and_train('CNN_3Layers', [
        tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(window_size, 1)),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])

    def forecast(model, series_normalized, window_size):
        predictions = []
        window_data = series_normalized[-window_size:]
        for i in range(len(test_normalized)):
            prediction = model.predict(window_data.reshape(1, window_size, 1), verbose=0)[0, 0]
            predictions.append(prediction)
            window_data = np.roll(window_data, shift=-1)
            window_data[-1] = test_normalized[i]
        return predictions

    region_predictions = {}
    for name, model in model_list:
        predictions = forecast(model, series_normalized, window_size)
        region_predictions[name] = predictions

    all_predictions[region] = region_predictions
    all_test_normalized[region] = test_normalized

process_region(region, file_path)


import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import tensorflow as tf
import os

region = 'ne'
file_path = '/content/drive/MyDrive/timeseries/splitted_dataset_region_ne.csv'
model_dir = '/content/drive/MyDrive/timeseries/trainnedmodels'
os.makedirs(model_dir, exist_ok=True)

all_predictions = {}
all_test_normalized = {}

def process_region(region, file_path):
    data = pd.read_csv(file_path, parse_dates=['din_instante'], index_col='din_instante')
    series = data['val_cargaenergiahomwmed'].values

    scalers = {'minmax': MinMaxScaler(), 'standard': StandardScaler(), 'robust': RobustScaler()}
    for scaler_name, scaler in scalers.items():
        series_normalized = scaler.fit_transform(series.reshape(-1, 1)).flatten()
        train_size = int(len(series_normalized) * 0.8)
        train_normalized = series_normalized[:train_size]
        test_normalized = series_normalized[train_size:]

        for window_size in [15, 30, 45]:
            X_train, y_train = [], []
            for i in range(len(train_normalized) - window_size):
                X_train.append(train_normalized[i:i + window_size])
                y_train.append(train_normalized[i + window_size])
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

            model_list = []

            def build_and_train(model_name, model_layers):
                model = tf.keras.Sequential(model_layers)
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_train, y_train, epochs=30, batch_size=64, verbose=0)
                model.save(os.path.join(model_dir, f"{model_name}_{scaler_name}_{window_size}.keras"))
                model_list.append((model_name, model))


            def build_and_train_BiLSTM_Attention(model_name):
                inputs = tf.keras.Input(shape=(window_size, 1))
                x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True))(inputs)
                attn_output = tf.keras.layers.Attention()([x, x])  # query = value = x
                x = tf.keras.layers.GRU(50)(attn_output)
                outputs = tf.keras.layers.Dense(1)(x)
                model = tf.keras.Model(inputs, outputs)
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_train, y_train, epochs=30, batch_size=64, verbose=0)
                model.save(os.path.join(model_dir, f"{model_name}_{scaler_name}_{window_size}.keras"))
                model_list.append((model_name, model))

            build_and_train('CNN_GRU_2Layers', [
                tf.keras.layers.Conv1D(64, 5, activation='relu', input_shape=(window_size, 1)),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Conv1D(64, 5, activation='relu'),
                tf.keras.layers.GRU(100, return_sequences=True),
                tf.keras.layers.GRU(50),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(1)
            ])

            build_and_train_BiLSTM_Attention('BiLSTM_Attention')

            def forecast(model, series_normalized, window_size):
                predictions = []
                window_data = series_normalized[-window_size:]
                for _ in range(len(test_normalized)):
                    prediction = model.predict(window_data.reshape(1, window_size, 1), verbose=0)[0, 0]
                    predictions.append(prediction)
                    window_data = np.roll(window_data, shift=-1)
                    window_data[-1] = prediction
                return predictions

            region_predictions = {}
            for name, model in model_list:
                preds = forecast(model, series_normalized, window_size)
                region_predictions[name] = preds

            all_predictions[f"{region}_{scaler_name}_{window_size}"] = region_predictions
            all_test_normalized[f"{region}_{scaler_name}_{window_size}"] = test_normalized

process_region(region, file_path)






import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import joblib
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import tensorflow as tf
import os


# Define file paths for each region
file_paths = {
    'se': '/content/drive/MyDrive/timeseries/splitted_dataset_region_se.csv',
    's': '/content/drive/MyDrive/timeseries/splitted_dataset_region_s.csv',
    'ne': '/content/drive/MyDrive/timeseries/splitted_dataset_region_ne.csv',
    'n': '/content/drive/MyDrive/timeseries/splitted_dataset_region_n.csv',
}

# Store predictions, actual values, and errors
all_predictions, all_test_normalized, all_errors = {}, {}, {}

def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    smape = np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))) * 100
    return mae, mse, rmse, mape, smape

def process_region(region, file_path):
    print(f"Processing {region}")
    data = pd.read_csv(file_path, parse_dates=['din_instante'], index_col='din_instante')
    series = data['val_cargaenergiahomwmed'].values

    train_size = int(len(series) * 0.8)
    train_series = series[:train_size]
    test_series = series[train_size:]

    model_dir = '/content/drive/MyDrive/timeseries/trainnedmodels'
    window_sizes = [15, 30, 45]
    scalers = {'minmax': MinMaxScaler(), 'standard': StandardScaler(), 'robust': RobustScaler()}

    region_predictions, region_errors = {}, {}

    for scaler_name, scaler in scalers.items():
        series_normalized = scaler.fit_transform(series.reshape(-1, 1)).flatten()
        test_normalized = series_normalized[train_size:]

        for window_size in window_sizes:
            print(f"Scaler: {scaler_name}, Window: {window_size}")

            def forecast(model, series_norm, test_norm, window_size):
                preds, window_data = [], series_norm[-window_size:]
                for _ in range(len(test_norm)):
                    prediction = model.predict(window_data.reshape(1, window_size, 1), verbose=0)[0, 0]
                    preds.append(prediction)
                    window_data = np.roll(window_data, shift=-1)
                    window_data[-1] = prediction
                return preds

            # Filter model files starting with a capital letter
            model_files = [
                f for f in os.listdir(model_dir)
                if f.endswith(f"{scaler_name}_{window_size}.keras") and f[0].isupper()
            ]

            for model_file in model_files:
                model_name = model_file.replace(f"_{scaler_name}_{window_size}.keras", "")
                model_path = os.path.join(model_dir, model_file)
                model = tf.keras.models.load_model(model_path)

                preds = forecast(model, series_normalized, test_normalized, window_size)
                mae, mse, rmse, mape, smape = calculate_metrics(test_normalized, preds)

                key = f"{model_name}_{scaler_name}_{window_size}"
                region_predictions[key] = preds
                region_errors[key] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'SMAPE': smape}


    all_predictions[region] = region_predictions
    all_test_normalized[region] = test_normalized
    all_errors[region] = region_errors

# Run for all regions
for region, file_path in file_paths.items():
    process_region(region, file_path)


# Print the calculated error metrics
for region, errors in all_errors.items():
    print(f"\nMétricas de erro para a região {region}:")
    for model, metrics in errors.items():
        print(f"Modelo: {model}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

# @title
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import joblib

# Define file paths for each region
file_paths = {
    'se': '/content/drive/MyDrive/timeseries/splitted_dataset_region_se.csv',
    's': '/content/drive/MyDrive/timeseries/splitted_dataset_region_s.csv',
    'ne': '/content/drive/MyDrive/timeseries/splitted_dataset_region_ne.csv',
    'n': '/content/drive/MyDrive/timeseries/splitted_dataset_region_n.csv',
}

# Store predictions, actual values, and errors
all_predictions = {}
all_test_normalized = {}
all_errors = {}

# Define moving average function
def moving_average(data, window_size):
    """Compute the moving average using a simple windowing technique."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Define function for calculating error metrics
def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    smape = np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))) * 100
    return mae, mse, rmse, mape, smape

# Define function to process each region
def process_region(region, file_path):
    # Load the data
    data = pd.read_csv(file_path, parse_dates=['din_instante'], index_col='din_instante')
    series = data['val_cargaenergiahomwmed'].values

    # Normalize the data
    scaler = MinMaxScaler()
    series_normalized = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    # Split into train and test sets
    train_size = int(len(series_normalized) * 0.8)
    train_normalized = series_normalized[:train_size]
    test_normalized = series_normalized[train_size:]

    # Define model names
    model_names = [
    'CNN', 'BiLSTM', 'GRU', 'CNN_BiLSTM', 'CNN_GRU', 'BiLSTM_GRU',
    'CNN_BiLSTM_GRU', 'CNN_2Layers', 'CNN_3Layers']


    # Define paths to load each model
    model_paths = {
    'CNN': f'/content/drive/MyDrive/timeseries/trainnedmodels/CNN.keras',
    'BiLSTM': f'/content/drive/MyDrive/timeseries/trainnedmodels/BiLSTM.keras',
    'GRU': f'/content/drive/MyDrive/timeseries/trainnedmodels/GRU.keras',
    'CNN_BiLSTM': f'/content/drive/MyDrive/timeseries/trainnedmodels/CNN_BiLSTM.keras',
    'CNN_GRU': f'/content/drive/MyDrive/timeseries/trainnedmodels/CNN_GRU.keras',
    'BiLSTM_GRU': f'/content/drive/MyDrive/timeseries/trainnedmodels/BiLSTM_GRU.keras',
    'CNN_BiLSTM_GRU': f'/content/drive/MyDrive/timeseries/trainnedmodels/CNN_BiLSTM_GRU.keras',
    'CNN_2Layers': f'/content/drive/MyDrive/timeseries/trainnedmodels/CNN_2Layers.keras',
    'CNN_3Layers': f'/content/drive/MyDrive/timeseries/trainnedmodels/CNN_3Layers.keras',
    }

    # Window size
    window_size = 30

    # Load models
    model_list = []
    for name in model_names:
        if 'keras' in model_paths[name]:
            model = tf.keras.models.load_model(model_paths[name])
        else:
            model = joblib.load(model_paths[name])
        model_list.append((name, model))

    # Forecast function for each model
    def forecast(model, series_normalized, window_size):
        predictions = []
        window_data = series_normalized[-window_size:]

        for i in range(len(test_normalized)):
            if isinstance(model, tf.keras.Sequential):
                prediction = model.predict(window_data.reshape(1, window_size, 1))[0, 0]
            else:
                prediction = model.predict(window_data.reshape(1, -1))[0]
            predictions.append(prediction)
            window_data = np.roll(window_data, shift=-1)
            window_data[-1] = test_normalized[i]

        return predictions

    # Store predictions and calculate metrics for each model
    region_predictions = {}
    region_errors = {}
    for name, model in model_list:
        print(name)
        if name != 'ARIMA':
            predictions = forecast(model, series_normalized, window_size)
        else:
            predictions = model.forecast(steps=len(test_normalized))

        region_predictions[name] = predictions
        mae, mse, rmse, mape, smape = calculate_metrics(test_normalized, predictions)
        region_errors[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'SMAPE': smape}

    # Save results for the region
    all_predictions[region] = region_predictions
    all_test_normalized[region] = test_normalized
    all_errors[region] = region_errors

# Process each region
for region, file_path in file_paths.items():
    process_region(region, file_path)

# Print the calculated error metrics
for region, errors in all_errors.items():
    print(f"\nMétricas de erro para a região {region}:")
    for model, metrics in errors.items():
        print(f"Modelo: {model}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")




