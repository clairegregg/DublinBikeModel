import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def read_daily_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    columns = ["TIME","AVAILABLE_BIKE_STANDS"]
    pre_pandemic = pd.read_csv("data/daily_pre-pandemic.csv", parse_dates=['TIME'], usecols=columns)
    pandemic = pd.read_csv("data/daily_pandemic.csv", parse_dates=['TIME'], usecols=columns)
    post_pandemic = pd.read_csv("data/daily_post-pandemic.csv", parse_dates=['TIME'], usecols=columns)
    return pre_pandemic, pandemic, post_pandemic

def new_lstm_dataset(dataset: [float], look_back: int = 1) -> np.ndarray:
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        dataX.append(dataset[i:(i+look_back),0])
        dataY.append(dataset[i+look_back, 0])
    return np.array(dataX), np.array(dataY)

def test_accuracy_lstm(dataset: [float], train_proportion: float):
    # Setup train test split
    train_size = int(len(dataset) * train_proportion)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    # Create features as one time step
    look_back=1
    trainX, trainY = new_lstm_dataset(train, look_back)
    testX, testY = new_lstm_dataset(test, look_back)

    # Reshape input to be [sample, time step, feature]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return

def main():
    # Ensure it is reproducable
    tf.random.set_seed(42)

    # Read data in
    pre_pandemic, pandemic, post_pandemic = read_daily_data()
    pre_pandemic_dataset = (pre_pandemic.drop(columns=['TIME'])).values
    pandemic_dataset = (pandemic.drop(columns=['TIME'])).values
    post_pandemic_dataset = (post_pandemic.drop(columns=['TIME'])).values

    # Normalise the datasets
    scaler = MinMaxScaler(feature_range=(0,1))
    pre_pandemic = scaler.fit_transform(pre_pandemic_dataset)
    pandemic = scaler.fit_transform(pandemic_dataset)
    post_pandemic = scaler.fit_transform(post_pandemic_dataset)
    

if __name__ == "__main__":
    main()