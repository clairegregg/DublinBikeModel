import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import math

def q_steps_ahead_pred(q: int, dd: int, lag: int, plot: bool, df: pd.DataFrame, test_prop: float):
    stride = 1

    t = df['TIME']
    y = df['AVAILABLE_BIKE_STANDS']
    XX = y[0:y.size-q-lag*dd:stride]
    for i in range(1,lag):
        X = y[i*dd:y.size-q-(lag-i)*dd:stride]
        XX = np.column_stack((XX,X))

    yy = y[lag*dd+q::stride].iloc[:]
    yy = yy.reset_index(drop=True)
    print(yy)
    tt = t[lag*dd+q::stride]

    train, test = train_test_split(np.arange(0,yy.size), test_size=test_prop)
    print(yy[train])
    model = Ridge(fit_intercept=False).fit(XX[train], yy[train])

    print(f"Intercept: {model.intercept_}, Coefficients: {model.coef_}")

    if plot:
        y_pred = model.predict(XX)
        plt.scatter(t,y, color='black', marker='.', label="Actual Data")
        plt.scatter(tt, y_pred, color='blue', marker='.', label="Predicted")
        plt.xlabel("Time")
        plt.ylabel("Bike usage")
        plt.legend()
        plt.show()

def combined_q_steps_ahead_pred(q: int, lag: int, plot: bool, df: pd.DataFrame, test_prop: float, dt: float):
    stride=1
    
    w = math.floor(4*7*24*60*60/dt)
    t = df['TIME']
    y = df['AVAILABLE_BIKE_STANDS']
    length = y.size-w-lag * w-q
    
    XX = y[q:q+length:stride]

    for i in range(1, lag):
        X=y[i*w+q:i*w+q+length:stride]
        XX= np.column_stack((XX,X))

    d = math.floor(7*24*60*60/dt)
    for i in range(0,lag):
        X=y[i*d+q:i*d+q+length:stride]
        XX=np.column_stack((XX,X))

    d = math.floor(24*60*60/dt)
    for i in range(0,lag):
        X=y[i*d+q:i*d+q+length:stride]
        XX=np.column_stack((XX,X))

    for i in range(0,lag):
        X=y[i:i+length:stride]
        XX=np.column_stack((XX,X))

    yy=y[lag*w+w+q:lag*w+w+q+length:stride]
    yy = yy.reset_index(drop=True)
    tt=t[lag*w+w+q:lag*w+w+q+length:stride]

    train,test = train_test_split(np.arange(0,yy.size), test_size=test_prop)
    model = Ridge(fit_intercept=False).fit(XX[train], yy[train])
    print(f"Intercept is {model.intercept_}, coefficient is {model.coef_}")

    if plot:
        y_pred = model.predict(XX)
        plt.scatter(t,y, color='black', marker='.', label="Actual Data")
        plt.scatter(tt, y_pred, color='blue', marker='.', label="Predicted")
        plt.xlabel("Time")
        plt.ylabel("Bike usage")
        plt.legend()
        plt.show()


def read_daily_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    pre_pandemic = pd.read_csv("data/daily_pre-pandemic.csv", parse_dates=['TIME'])
    pandemic = pd.read_csv("data/daily_pandemic.csv", parse_dates=['TIME'])
    post_pandemic = pd.read_csv("data/daily_post-pandemic.csv", parse_dates=['TIME'])
    return pre_pandemic, pandemic, post_pandemic

def main():
    pre_pandemic, pandemic, post_pandemic = read_daily_data()
    t_full=pd.array(pd.DatetimeIndex(pre_pandemic.iloc[:,1]).astype(np.int64))/1000000000
    dt = t_full[1]-t_full[0]

    combined_q_steps_ahead_pred(10, 3, True, pre_pandemic, 0.2, dt)

if __name__ == "__main__":
    main()