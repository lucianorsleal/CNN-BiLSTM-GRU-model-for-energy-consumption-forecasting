from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# Output directory for models
region='ne'
file_path ='/content/drive/MyDrive/timeseries/splitted_dataset_region_ne.csv'
model_dir = '/content/drive/MyDrive/timeseries/trainnedmodels'


os.makedirs(model_dir, exist_ok=True)

# Store predictions and actual values
all_predictions = {}
all_test_normalized = {}

def process_region( region, file_path):
    cnn_lstm_params = (64, 3, 50, 'tanh', 'adam', 64)
    cnn_params = (64, 3, 'relu', 'adam', 64)
    lstm_params = (200, 0.2, 'relu', 'adam', 64)
    gru_params = (200, 0.2, 'relu', 'adam', 64)
    rf_params = (300, 10, 10, 5, None)
    svr_params = (100, 0.1, 'linear', 'scale')

    data = pd.read_csv(file_path, parse_dates=['din_instante'], index_col='din_instante')
    series = data['val_cargaenergiahomwmed'].values

    scaler = MinMaxScaler()
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

    # CNN+LSTM
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=cnn_lstm_params[0], kernel_size=cnn_lstm_params[1], activation=cnn_lstm_params[3], input_shape=(window_size, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=cnn_lstm_params[0], kernel_size=cnn_lstm_params[1], activation=cnn_lstm_params[3]),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.LSTM(cnn_lstm_params[2], return_sequences=False),
        tf.keras.layers.Dense(50, activation=cnn_lstm_params[3]),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=cnn_lstm_params[4], loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=cnn_lstm_params[5], verbose=1)
    model_path = os.path.join(model_dir, "cnn_lstm.keras")
    model.save(model_path)
    model_list.append(('CNN+LSTM', model))

    # CNN
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=cnn_params[0], kernel_size=cnn_params[1], activation=cnn_params[2], input_shape=(window_size, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=cnn_params[0], kernel_size=cnn_params[1], activation=cnn_params[2]),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation=cnn_params[2]),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=cnn_params[3], loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=cnn_params[4], verbose=1)
    model_path = os.path.join(model_dir, "cnn.keras")
    model.save(model_path)
    model_list.append(('CNN', model))

    # LSTM
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(lstm_params[0], activation=lstm_params[2], input_shape=(window_size, 1), return_sequences=False),
        tf.keras.layers.Dropout(lstm_params[1]),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=lstm_params[3], loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=lstm_params[4], verbose=1)
    model_path = os.path.join(model_dir, "lstm.keras")
    model.save(model_path)
    model_list.append(('LSTM', model))



    # GRU
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(gru_params[0], activation=gru_params[2], input_shape=(window_size, 1), return_sequences=False),
        tf.keras.layers.Dropout(gru_params[1]),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=gru_params[3], loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=gru_params[4], verbose=1)
    model.save(os.path.join(model_dir, "gru.keras"))
    model_list.append(('GRU', model))

    # BiGRU
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_params[0], activation=gru_params[2], return_sequences=False), input_shape=(window_size, 1)),
        tf.keras.layers.Dropout(gru_params[1]),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=gru_params[3], loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=gru_params[4], verbose=1)
    model.save(os.path.join(model_dir, "bigru.keras"))
    model_list.append(('BiGRU', model))


    # LSTM+GRU
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(lstm_params[0], activation=lstm_params[2], return_sequences=True, input_shape=(window_size, 1)),
        tf.keras.layers.GRU(gru_params[0], activation=gru_params[2], return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)
    model.save(os.path.join(model_dir, "lstm_gru.keras"))
    model_list.append(('LSTM+GRU', model))

    # CNN+LSTM+GRU
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.GRU(50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)
    model.save(os.path.join(model_dir, "cnn_lstm_gru.keras"))
    model_list.append(('CNN+LSTM+GRU', model))

    # BiLSTM
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_params[0], activation=lstm_params[2], return_sequences=False), input_shape=(window_size, 1)),
        tf.keras.layers.Dropout(lstm_params[1]),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=lstm_params[3], loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=lstm_params[4], verbose=1)
    model.save(os.path.join(model_dir, "bilstm.keras"))
    model_list.append(('BiLSTM', model))

    # CNN+BiLSTM
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=False)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)
    model.save(os.path.join(model_dir, "cnn_bilstm.keras"))
    model_list.append(('CNN+BiLSTM', model))

    # CNN+BiLSTM+GRU
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True)),
        tf.keras.layers.GRU(50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)
    model.save(os.path.join(model_dir, "cnn_bilstm_gru.keras"))
    model_list.append(('CNN+BiLSTM+GRU', model))


    # SVR
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    svr_model = SVR(kernel=svr_params[2], C=svr_params[0], epsilon=svr_params[1], gamma=svr_params[3])
    svr_model.fit(X_train_flat, y_train)
    model_path = os.path.join(model_dir, "svr.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(svr_model, f)
    model_list.append(('SVR', svr_model))

    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=rf_params[0],
        max_depth=rf_params[1],
        min_samples_split=rf_params[2],
        min_samples_leaf=rf_params[3],
        max_features=rf_params[4]
    )
    rf_model.fit(X_train_flat, y_train)
    model_path = os.path.join(model_dir, "rf.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(rf_model, f)
    model_list.append(('Random Forest', rf_model))

    def forecast(model, series_normalized, window_size):
        predictions = []
        window_data = series_normalized[-window_size:]

        for i in range(len(test_normalized)):
            if isinstance(model, tf.keras.Model):
                prediction = model.predict(window_data.reshape(1, window_size, 1), verbose=0)[0, 0]
            else:
                prediction = model.predict(window_data.reshape(1, -1))[0]
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
    'CNN+LSTM', 'CNN', 'LSTM', 'SVR', 'Random Forest', 'ARIMA',
    'GRU', 'BiGRU', 'LSTM+GRU', 'CNN+LSTM+GRU', 'BiLSTM', 'CNN+BiLSTM', 'CNN+BiLSTM+GRU']


    # Define paths to load each model
    model_paths = {
    'CNN+LSTM': f'/content/drive/MyDrive/timeseries/trainnedmodels/cnn_lstm.keras',
    'CNN': f'/content/drive/MyDrive/timeseries/trainnedmodels/cnn.keras',
    'LSTM': f'/content/drive/MyDrive/timeseries/trainnedmodels/lstm.keras',
    'SVR': f'/content/drive/MyDrive/timeseries/trainnedmodels/svr.pkl',
    'Random Forest': f'/content/drive/MyDrive/timeseries/trainnedmodels/rf.pkl',
    'ARIMA': f'/content/drive/MyDrive/timeseries/trainnedmodels/arima-model.pkl',
    'GRU': f'/content/drive/MyDrive/timeseries/trainnedmodels/gru.keras',
    'BiGRU': f'/content/drive/MyDrive/timeseries/trainnedmodels/bigru.keras',
    'LSTM+GRU': f'/content/drive/MyDrive/timeseries/trainnedmodels/lstm_gru.keras',
    'CNN+LSTM+GRU': f'/content/drive/MyDrive/timeseries/trainnedmodels/cnn_lstm_gru.keras',
    'BiLSTM': f'/content/drive/MyDrive/timeseries/trainnedmodels/bilstm.keras',
    'CNN+BiLSTM': f'/content/drive/MyDrive/timeseries/trainnedmodels/cnn_bilstm.keras',
    'CNN+BiLSTM+GRU': f'/content/drive/MyDrive/timeseries/trainnedmodels/cnn_bilstm_gru.keras',
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

import matplotlib.pyplot as plt
import os

num_points = 60
# Define a mapping for region keys to their names
region_names = {
    'n': 'North',
    'ne': 'Northeast',
    's': 'South',
    'se': 'Southeast'
}

# Plotting results after processing all regions
for region, predictions in all_predictions.items():
    plt.figure(figsize=(15, 6))
    window_size = 11  # Adjust for moving average smoothing

    # Get the test normalized values for this region
    test_normalized = all_test_normalized[region]

    for model_name, preds in predictions.items():
        # Apply moving average to predictions
        smoothed_preds = moving_average(preds, window_size)
        # Plot only the first 60 points
        plt.plot(range(num_points), smoothed_preds[:num_points], label=model_name)

    # Apply moving average to the actual values
    smoothed_actual = moving_average(test_normalized, window_size)
    plt.plot(range(num_points), smoothed_actual[:num_points], label='Actual', color='black')

    # Set title with the mapped region name
    plt.title(f"{region_names[region]}")

    plt.xlabel("Time")
    plt.ylabel("Normalized Value")
    plt.legend()

    output_dir = '/content/drive/MyDrive/timeseries/plots/'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}{region}_predictoins_smoothed.pdf', format='pdf')
    plt.show()
    plt.close()


# Plotting results after processing all regions
for region, predictions in all_predictions.items():
    plt.figure(figsize=(18, 6))
    window_size = 3  # Adjust for moving average smoothing

    for model_name, preds in predictions.items():
        # Apply moving average to predictions
        smoothed_preds = moving_average(preds, window_size)
        # Plot only the first 30 points
        plt.plot(range(30), smoothed_preds[:30], label=model_name)

    # Apply moving average to the actual values (you may need to track actuals as well)
    smoothed_actual = moving_average(test_normalized, window_size)
    plt.plot(range(30), smoothed_actual[:30], label='Actual', color='black')

    plt.title(f"Region: {region} - Model Predictions (Smoothed)")
    plt.xlabel("Time")
    plt.ylabel("Normalized Value")
    plt.legend()

    output_dir = '/content/drive/MyDrive/timeseries/plots/'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}{region}_predictions_smoothed.svg', format='svg')
    plt.close()

!pip install scikit-posthocs

import numpy as np
import scipy.stats as stats
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt

# MAE values for each model across 4 regions (rows = regions, columns = models)
mae_values = np.array([
    [0.0418, 0.0538, 0.0502, 0.0512, 0.0574, 0.1714, 0.0479, 0.0444, 0.0487, 0.0483, 0.0484, 0.0507, 0.0336],
    [0.0520, 0.0634, 0.0725, 0.0644, 0.0774, 0.1786, 0.0718, 0.0662, 0.0688, 0.0563, 0.0730, 0.0668, 0.0503],
    [0.0425, 0.0589, 0.0487, 0.0534, 0.0689, 0.2048, 0.0479, 0.0458, 0.0504, 0.0492, 0.0477, 0.0516, 0.0348],
    [0.0394, 0.0668, 0.0489, 0.0449, 0.0900, 0.2911, 0.0488, 0.0499, 0.0535, 0.0559, 0.0509, 0.0511, 0.0368]
])

# Model names
models = [
    'CNN+LSTM', 'CNN', 'LSTM', 'SVR', 'RF', 'ARIMA', 'GRU', 'BiGRU',
    'LSTM+GRU', 'CNN+LSTM+GRU', 'BiLSTM', 'CNN+BiLSTM', 'CNN+BiLSTM+GRU'
]

# Perform Friedman test
friedman_stat, p_value = stats.friedmanchisquare(*[mae_values[:, i] for i in range(mae_values.shape[1])])
print(f"Friedman chi-square statistic = {friedman_stat:.4f}, p-value = {p_value:.4f}")

# If significant, perform Nemenyi post-hoc test
if p_value < 0.05:
    print("Significant differences detected. Proceeding with Nemenyi post-hoc test.")

    # Compute Nemenyi test using scikit-posthocs
    nemenyi_result = sp.posthoc_nemenyi_friedman(mae_values)

    # Set up the heatmap plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(nemenyi_result, annot=True, fmt=".3f", xticklabels=models, yticklabels=models, cmap="coolwarm_r")
    plt.title("Nemenyi Post-hoc Test — Pairwise p-values")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

else:
    print("No significant differences found between models.")


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# MAE values for each model across 4 regions
mae_values = np.array([
    [0.0418, 0.0538, 0.0502, 0.0512, 0.0574, 0.1714, 0.0479, 0.0444, 0.0487, 0.0483, 0.0484, 0.0507, 0.0336],
    [0.0520, 0.0634, 0.0725, 0.0644, 0.0774, 0.1786, 0.0718, 0.0662, 0.0688, 0.0563, 0.0730, 0.0668, 0.0503],
    [0.0425, 0.0589, 0.0487, 0.0534, 0.0689, 0.2048, 0.0479, 0.0458, 0.0504, 0.0492, 0.0477, 0.0516, 0.0348],
    [0.0394, 0.0668, 0.0489, 0.0449, 0.0900, 0.2911, 0.0488, 0.0499, 0.0535, 0.0559, 0.0509, 0.0511, 0.0368]
])

# Model names
models = [
    'CNN+LSTM', 'CNN', 'LSTM', 'SVR', 'RF', 'ARIMA', 'GRU', 'BiGRU',
    'LSTM+GRU', 'CNN+LSTM+GRU', 'BiLSTM', 'CNN+BiLSTM', 'CNN+BiLSTM+GRU'
]

# Compute average ranks: lower MAE = better rank
ranks = np.argsort(np.argsort(mae_values, axis=1), axis=1) + 1  # rank within each row (region)
avg_ranks = np.mean(ranks, axis=0)

# Sort models by average rank (for nicer plotting)
sorted_indices = np.argsort(avg_ranks)
sorted_models = [models[i] for i in sorted_indices]
sorted_avg_ranks = avg_ranks[sorted_indices]

# Friedman test
friedman_stat, p_value = stats.friedmanchisquare(*[mae_values[:, i] for i in range(mae_values.shape[1])])
print(f"Friedman chi-square statistic = {friedman_stat:.4f}, p-value = {p_value:.4f}")

# Plot horizontal bar chart of average ranks (like the PDF)
plt.figure(figsize=(10, 6))
plt.scatter(sorted_avg_ranks, sorted_models, color='royalblue', s=100, zorder=3)
plt.hlines(sorted_models, xmin=min(sorted_avg_ranks)-0.1, xmax=max(sorted_avg_ranks)+0.1, color='lightgray', linestyle='dotted', zorder=1)
plt.xlabel("Average Rank (Lower is Better)")
plt.title("Average Ranks of Models (Friedman Test Ranking)")

# Optional: grid and labels
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.xlim(min(sorted_avg_ranks)-0.2, max(sorted_avg_ranks)+0.2)
plt.tight_layout()
plt.savefig('friedman.pdf')
plt.show()



