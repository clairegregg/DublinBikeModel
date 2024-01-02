import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import math

def one_step_ahead(y: pd.Series, lag: int, dt: float) -> float:
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

    model = Ridge(fit_intercept=False).fit(XX, yy)

    return model.predict(XX)[-1]

def read_daily_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    pre_pandemic = pd.read_csv("data/daily_pre-pandemic.csv", parse_dates=['TIME'])
    pandemic = pd.read_csv("data/daily_pandemic.csv", parse_dates=['TIME'])
    post_pandemic = pd.read_csv("data/daily_post-pandemic.csv", parse_dates=['TIME'])
    return pre_pandemic, pandemic, post_pandemic

def main():
    pre_pandemic, pandemic, post_pandemic = read_daily_data()
    t_full=pd.array(pd.DatetimeIndex(pre_pandemic.iloc[:,1]).astype(np.int64))/1000000000
    dt = t_full[1]-t_full[0]
    y = pre_pandemic['AVAILABLE_BIKE_STANDS']

    for _ in range(pandemic.shape[0]):
        new_y = one_step_ahead(y, 3, dt)
        y[len(y)] = new_y

    print(len(pandemic['TIME']))
    print(len(y[pre_pandemic.shape[0]:]))
    print(len(y))
    print(pre_pandemic.shape[0])
    plt.scatter(pd.concat([pre_pandemic['TIME'], pandemic['TIME']]), pd.concat([pre_pandemic["AVAILABLE_BIKE_STANDS"], pandemic["AVAILABLE_BIKE_STANDS"]]), color='black', marker='.', label="Actual Data")
    plt.scatter(pandemic['TIME'], y[pre_pandemic.shape[0]:], color='blue', marker='.', label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Bike usage")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()