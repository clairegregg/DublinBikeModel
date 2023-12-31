import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt, floor
from tqdm import tqdm

def read_daily_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    columns = ["TIME","AVAILABLE_BIKE_STANDS"]
    pre_pandemic = pd.read_csv("data/daily_pre-pandemic.csv", parse_dates=['TIME'], usecols=columns)
    pandemic = pd.read_csv("data/daily_pandemic.csv", parse_dates=['TIME'], usecols=columns)
    post_pandemic = pd.read_csv("data/daily_post-pandemic.csv", parse_dates=['TIME'], usecols=columns)
    return pre_pandemic, pandemic, post_pandemic

def baseline_model(train: [float], test: [float]):
    history = [x for x in train]
    predictions = []

    for i in range (len(test)):
        # Make prediction
        predictions.append(history[-1])

        # New observation
        history.append(test[i])
    
    # Report performance
    error = sqrt(mean_squared_error(test, predictions))
    print("Baseline model mean squared error: %.2f"%error)

    # Plot observed vs predicted
    plt.plot(test)
    plt.plot(predictions)
    plt.show()

def difference_series(dataset: [float], interval: int=1) -> pd.Series:
    diff = list()
    for i in tqdm(range(interval, len(dataset)), desc="Making dataset stationary"):
        value = dataset[i] - dataset[i-interval]
        diff.append(value)
    return pd.Series(diff)

def inverse_difference_series(history, yhat, interval=1):
 return yhat + history[-interval]

def series_to_input_output(data: [float], lag: int=1) -> pd.DataFrame:
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0,inplace=True)
    return df

def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(train)

    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)

    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)

    return scaler, train_scaled, test_scaled

def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0,-1]

def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])

    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer='adam')
    
    for _ in tqdm(range(nb_epoch), desc="Fitting model"):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model

def forecast(model, batch_size, row):
    X = row[0:-1]
    X = X.reshape(1, 1, len(X))
    yhat = model.predicted(X, batch_size=batch_size)
    return yhat[0,0]


def main():
    # Ensure it is reproducable
    tf.random.set_seed(42)

    # Read data in
    pre_pandemic, pandemic, post_pandemic = read_daily_data()
    print(pre_pandemic)
    pre_pandemic_dataset = (pre_pandemic.drop(columns=['TIME'])).values

    # Transform to be stationary
    pre_pandemic_diff = difference_series(pre_pandemic_dataset, 1)

    # Transform to supervised learning (input/output)
    pre_pandemic_supervised = series_to_input_output(pre_pandemic_diff).values

    # Split into train/test
    train_size = floor(0.8* len(pre_pandemic_supervised))
    train, test = pre_pandemic_supervised[0:train_size], pre_pandemic_supervised[train_size:]

    # Normalise the data
    scaler, train_scaled, test_scaled = scale(train, test)

    # Fit the model
    model = fit_lstm(train_scaled, 1, 3000, 4)

    # Forecast the full training dataset to build up state for forecasting
    train_reshaped = train_scaled[:,0].reshape(len(train_scaled), 1, 1)
    model.predict(train_reshaped, batch_size=1)

    # walk-forward validation on the test data
    predictions = list()
    for i in tqdm(range(len(test_scaled)), desc=f"Making predictions"):
        # make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forecast(model, 1, X)
        # invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # invert differencing
        yhat = inverse_difference_series(pre_pandemic_dataset, yhat, len(test_scaled)+1-i)
        # store forecast
        predictions.append(yhat)
        expected = pre_pandemic_dataset[len(train) + i + 1]
    print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
    
    # report performance
    rmse = sqrt(mean_squared_error(pre_pandemic_dataset[train_size:], predictions))
    print('Test RMSE: %.3f' % rmse)
    # line plot of observed vs predicted
    plt.plot(pre_pandemic_dataset[train_size:])
    plt.plot(predictions)
    plt.show()

    

if __name__ == "__main__":
    main()