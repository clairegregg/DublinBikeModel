import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import math

def one_step_ahead(y: pd.Series, lag: int, dt: float, alpha: float) -> float:
    stride, q = 1, 1

    # Number of samples in a month
    m = math.floor(4*7*24*60*60/dt)
    length = y.size-m-lag * m-q

    # Initialise model features
    XX = y[q:q+length:stride]

    for i in range(1, lag):
        X=y[i*m+q:i*m+q+length:stride]
        XX= np.column_stack((XX,X))

    # Number of samples in a week
    w = math.floor(7*24*60*60/dt)
    for i in range(0,lag):
        X=y[i*w+q:i*w+q+length:stride]
        XX=np.column_stack((XX,X))

    # Number of samples in a day
    d = math.floor(24*60*60/dt)
    for i in range(0,lag):
        X=y[i*d+q:i*d+q+length:stride]
        XX=np.column_stack((XX,X))

    # Short term trend
    for i in range(0,lag):
        X=y[i:i+length:stride]
        XX=np.column_stack((XX,X))

    # Format output feature
    yy=y[lag*w+w+q:lag*w+w+q+length:stride]
    yy = yy.reset_index(drop=True)

    model = Ridge(fit_intercept=False, alpha=alpha).fit(XX, yy)

    return model.predict(XX)[-1]

def multi_step_prediction(train: pd.DataFrame, test: pd.DataFrame, lag: int, dt: float, alpha: float, plot: bool):
    y = train.copy()['AVAILABLE_BIKE_STANDS']
    for _ in range(len(test)):
        new_y = one_step_ahead(y, lag, dt, alpha)
        y[len(y)] = new_y

    y_pred = y[len(train):]
    y_true = test['AVAILABLE_BIKE_STANDS']

    mse = mean_squared_error(y_true, y_pred)

    if plot:
        full_x = pd.concat([train['TIME'], test['TIME']])
        full_y = pd.concat([train["AVAILABLE_BIKE_STANDS"], test["AVAILABLE_BIKE_STANDS"]])
        plt.plot(full_x, full_y, color='black', label="Actual Data")
        plt.plot(test['TIME'], y_pred, color='blue', label="Predicted")
        plt.xlabel("Time")
        plt.ylabel("Bike usage")
        plt.legend()
        plt.show()

    return mse

def cross_validation_on_pre_pandemic(pre_pandemic: pd.DataFrame, dt: float):
    c_vals = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    lags = np.arange(start=1, stop=15)
    current_min_error = np.inf
    current_min_c_val, current_min_lag = 0, 0

    train_size = math.floor(0.8*(pre_pandemic.shape[0]))
    train, test = pre_pandemic[0:train_size], pre_pandemic[train_size:]

    for c in c_vals:
        for lag in lags:
            alpha = 1/(2*c)
            err = multi_step_prediction(train, test, lag, dt, alpha, False)
            print("checked 1")
            if err < current_min_error:
                current_min_error = err
                current_min_c_val = c
                current_min_lag = lag

    print(f"Lowest mean squared error with c={current_min_c_val} and lag={current_min_lag}. Error is {current_min_error}.")
    multi_step_prediction(train, test, current_min_lag, dt, 1/(2*c), True)


def read_daily_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    pre_pandemic = pd.read_csv("data/daily_pre-pandemic.csv", parse_dates=['TIME'])
    pandemic = pd.read_csv("data/daily_pandemic.csv", parse_dates=['TIME'])
    post_pandemic = pd.read_csv("data/daily_post-pandemic.csv", parse_dates=['TIME'])
    return pre_pandemic, pandemic, post_pandemic

def main():
    pre_pandemic, pandemic, post_pandemic = read_daily_data()
    t_full=pd.array(pd.DatetimeIndex(pre_pandemic.iloc[:,1]).astype(np.int64))/1000000000
    dt = t_full[1]-t_full[0]

    cross_validation_on_pre_pandemic(pre_pandemic, dt)

if __name__ == "__main__":
    main()