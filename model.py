import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def one_step_ahead(y: pd.Series, lag: int, dt: float, alpha: float, trends: [str], stride: int) -> float:
    q = 1

    # Initialise sample numbers for different periods
    hy = math.floor(27*7*24*60*60/dt) # Number of samples in a half year
    m = math.floor(4*7*24*60*60/dt) # Number of samples in a month
    w = math.floor(7*24*60*60/dt) # Number of samples in a week
    d = math.floor(24*60*60/dt) # Number of samples in a day (1)

    # Length depending on what trends are being trained
    x = hy if "half-year" in trends else (m if "month" in trends else (w if "week" in trends else d))
    length = y.size-x-(lag*x)-q 
    
    # If the length of the feature is less than 0 or no period of time is being used for training, return 
    if length <= 0:
        return np.nan
    if len(trends) == 0:
        return np.nan


    # Write model features
    XX = y[q:q+length:stride]
    if "half-year" in trends:
        for i in range(0, lag):
            X=y[i*hy+q:i*hy+q+length:stride]
            XX= np.column_stack((XX,X))

    if "month" in trends:
        for i in range(0, lag):
            X=y[i*m+q:i*m+q+length:stride]
            XX= np.column_stack((XX,X))

    if "week" in trends:
        for i in range(0,lag):
            X=y[i*w+q:i*w+q+length:stride]
            XX=np.column_stack((XX,X))

    if "short-term" in trends:
        for i in range(0,lag):
            X=y[i:i+length:stride]
            XX=np.column_stack((XX,X))

    # Format output feature
    yy=y[lag*x+x+q:lag*x+x+q+length:stride]
    yy = yy.reset_index(drop=True)

    # Train model
    model = Ridge(fit_intercept=False, alpha=alpha, random_state=42).fit(XX, yy)

    # Return model prediction for next bike usage statistic
    return model.predict(XX)[-1]

def multi_step_prediction(train: pd.DataFrame, test: pd.DataFrame, lag: int, dt: float, alpha: float, trends: [str], stride: int, plot: bool, check_error: bool, model_post_pandemic: bool):
    y = train.copy()['AVAILABLE_BIKE_STANDS']
    curr_date = test['TIME'].iloc[0].date()
    days_to_predict = (pd.to_datetime('2030-01-01').date()-curr_date).days
    predict_days = pd.Series()
    predict_len = days_to_predict if model_post_pandemic else len(test)
    for _ in range(predict_len):
        new_y = one_step_ahead(y, lag, dt, alpha, trends, stride)
        if np.isnan(new_y):
            return np.inf
        y[len(y)] = new_y
        predict_days[len(predict_days)] = curr_date
        curr_date = curr_date + timedelta(days=1)

    mse = np.inf
    y_pred = y[len(train):]
    if check_error:
        y_true = test['AVAILABLE_BIKE_STANDS']
        mse = mean_squared_error(y_true, y_pred)
    
    if model_post_pandemic:
        curr_date = test['TIME'].iloc[-1].date() + timedelta(days=1)
        days_to_predict = (pd.to_datetime('2030-01-01').date()-curr_date).days
        y_pandemic = pd.concat([train['AVAILABLE_BIKE_STANDS'], test['AVAILABLE_BIKE_STANDS']])
        pandemic_dates = pd.Series()
        for _ in range(days_to_predict):
            new_y = one_step_ahead(y_pandemic, lag, dt, alpha, trends, stride)
            y_pandemic[len(y_pandemic)] = new_y
            pandemic_dates[len(pandemic_dates)] = curr_date
            curr_date = curr_date + timedelta(days=1)

        y_pandemic_pred = y_pandemic[(len(train) + len(test)):]
    
    if plot:
        test = test.set_index(test['TIME'])
        pandemic = test[:'2022-01-28']
        post_pandemic = test['2022-01-28':]
        plt.plot(train['TIME'], train["AVAILABLE_BIKE_STANDS"], color='blue', label="Actual pre-pandemic bike usage")
        if post_pandemic.shape[0] == 0:
            plt.plot(test['TIME'], test["AVAILABLE_BIKE_STANDS"], color='green', label="Actual pandemic bike usage")
        else:
            plt.plot(pandemic['TIME'], pandemic["AVAILABLE_BIKE_STANDS"], color='green', label="Actual pandemic bike usage")
            plt.plot(post_pandemic['TIME'], post_pandemic["AVAILABLE_BIKE_STANDS"], color='red', label="Actual post-pandemic bike usage")

        plt.plot(predict_days, y_pred, color='lime', label="Predicted bike usage (without pandemic)")

        if model_post_pandemic:
            plt.plot(pandemic_dates, y_pandemic_pred, color='fuchsia', label="Predicted bike usage (given pandemic)")
        plt.xlabel("Time")
        plt.ylabel("Bike usage")
        #plt.suptitle(f"Bike usage time series model, error = {mse}")          
        #plt.title(f"lag={lag}, alpha={alpha}, stride={stride}, modelled trends={trends}")
        plt.suptitle("Predicted bike usage given no pandemic")
        plt.title(f"using Ridge regression multistep time prediction \n lag={lag}, alpha={alpha}, stride={stride}, trends={trends}", fontsize=10)
        plt.legend()
        plt.show()

    return mse

def cross_validation_on_pre_pandemic(pre_pandemic: pd.DataFrame, dt: float):
    c_vals = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    lags = np.arange(start=1, stop=30)
    trends = ["short-term", "week", "month", "half-year"]
    strides = np.arange(start=1, stop=10)
    trend_combinations = [trends[i:j] for i in range(len(trends)) for j in range(i + 1, len(trends) + 1)]
    current_min_error = np.inf
    current_min_c_val, current_min_lag, current_min_stride = 0, 0, 0
    current_min_trends = []

    train_size = math.floor(0.8*(pre_pandemic.shape[0]))
    train, test = pre_pandemic[0:train_size], pre_pandemic[train_size:]

    for c in tqdm(c_vals, desc="Cross-validation"):
        for lag in lags:
            for trend_combination in trend_combinations:
                for stride in strides:
                    alpha = (1/(2*c))
                    err = multi_step_prediction(train, test, lag, dt, alpha, trend_combination, stride, False)
                    if err < current_min_error:
                        print(f"{c}, {lag}, {stride},  {trend_combination}, {err}")
                        current_min_error = err
                        current_min_c_val = c
                        current_min_lag = lag
                        current_min_trends = trend_combination
                        current_min_stride = stride

    print(f"Lowest mean squared error with c={current_min_c_val}, lag={current_min_lag}, stride={current_min_stride}, and tracking trends {current_min_trends}. Error is {current_min_error}.")
    multi_step_prediction(train=train, test=test, lag=current_min_lag, dt=dt, alpha=(1/(2*current_min_c_val)), trends=current_min_trends, stride=current_min_stride, plot=True)

def add_one_year(input_datetime):
    new_datetime = input_datetime + relativedelta(years=1)
    return new_datetime

def statistical_prediction(pre_pandemic_df: pd.DataFrame, pandemic_df: pd.DataFrame, post_pandemic_df: pd.DataFrame, include_post_pandemic: bool, date_to_predict_to: str):
    start_year_index = pre_pandemic_df[pre_pandemic_df['TIME'].dt.date == pd.to_datetime('2019-01-01').date()].index[0]
    end_year_index = pre_pandemic_df[pre_pandemic_df['TIME'].dt.date == pd.to_datetime('2020-01-01').date()].index[0]

    year_df = pre_pandemic_df[start_year_index:end_year_index].copy().reset_index(drop=True)

    year = year_df['AVAILABLE_BIKE_STANDS']
    time = year_df['TIME']

    days_to_predict = (pd.to_datetime(date_to_predict_to).date()-pd.to_datetime('2020-01-01').date()).days

    num_extra_years = math.floor(days_to_predict/365)
    usage = pd.Series()
    date_series = pd.Series()

    for _ in range(num_extra_years):
        usage = pd.concat([usage, year])
        date_series = pd.concat([date_series, (time.apply(add_one_year))])
        time = time.apply(add_one_year)
    
    num_extra_days = days_to_predict % 365
    for index in range(num_extra_days):
        usage[len(usage)] = year[index]
        date_series[len(date_series)] = add_one_year(time[index])

    plt.plot(pre_pandemic_df['TIME'], pre_pandemic_df['AVAILABLE_BIKE_STANDS'], color='blue', label="Actual pre-pandemic bike usage")
    plt.plot(pandemic_df['TIME'], pandemic_df['AVAILABLE_BIKE_STANDS'], color='green', label="Actual pandemic bike usage")
    if include_post_pandemic:
        plt.plot(post_pandemic_df['TIME'], post_pandemic_df['AVAILABLE_BIKE_STANDS'], color='red', label="Actual post-pandemic bike usage")
    plt.plot(date_series, usage, color='lime', label="Predicted bike usage without pandemic")
    plt.xlabel("Time")
    plt.ylabel("Bike usage")
    plt.suptitle("Predicted bike usage given no pandemic")
    plt.title(f"using statistical prediction (copying pattern from 2019)", fontsize=10)
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

    #cross_validation_on_pre_pandemic(pre_pandemic, dt)

    multi_step_prediction(train=pre_pandemic, test=pandemic, lag=14, dt=dt, alpha=1/(2*0.0001), trends=["short-term", "week", "month"], stride=3, plot=True, check_error=False, model_post_pandemic=False)
    multi_step_prediction(train=pre_pandemic, test=pandemic, lag=1, dt=dt, alpha=1/(2*0.0001), trends=["short-term", "week", "month", "half-year"], stride=3, plot=True, check_error=False, model_post_pandemic=False)
    statistical_prediction(pre_pandemic_df=pre_pandemic, pandemic_df=pandemic, post_pandemic_df=None, include_post_pandemic= False, date_to_predict_to=pandemic['TIME'].iloc[-1])
    
    multi_step_prediction(train=pre_pandemic, test=pd.concat([pandemic, post_pandemic]), lag=14, dt=dt, alpha=1/(2*0.0001), trends=["short-term", "week", "month"], stride=3, plot=True, check_error=False, model_post_pandemic=True)
    multi_step_prediction(train=pre_pandemic, test=pd.concat([pandemic, post_pandemic]), lag=1, dt=dt, alpha=1/(2*0.0001), trends=["short-term", "week", "month", "half-year"], stride=3, plot=True, check_error=False, model_post_pandemic=True)
    statistical_prediction(pre_pandemic_df=pre_pandemic, pandemic_df=pandemic, post_pandemic_df=post_pandemic, include_post_pandemic=True, date_to_predict_to='2030-01-01')


if __name__ == "__main__":
    main()