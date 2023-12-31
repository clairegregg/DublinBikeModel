import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Given:
#   - Series of bike usage values over time
#   - The lag to calculate with
#   - The time between data points
#   - The alpha hyperparameter for a Ridge regression model
#   - The types of trends to track - any combination of [short term, week, month, half-year]
#   - The stride calculated with
# Predict the next bike usage data point
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

# Given train and test datasets, predict the time series following the training period.
# If check_error is True, calculate the mean squared error between the test dataset and the predicted time series
# If model_post_pandemic is true, predict up until 2030
def multi_step_prediction(train: pd.DataFrame, test: pd.DataFrame, lag: int, dt: float, alpha: float, trends: [str], stride: int, plot: bool, check_error: bool, model_post_pandemic: bool):
    # Find how many days to predict - if modelling post-pandemic, predict up until 2030
    curr_date = test['TIME'].iloc[0].date()
    days_to_predict = (pd.to_datetime('2030-01-01').date()-curr_date).days if model_post_pandemic else len(test)
    
    # Initialise training data and location for storing values
    y = train.copy()['AVAILABLE_BIKE_STANDS']
    predict_days = pd.Series()

    # Predict all days required
    for _ in range(days_to_predict):
        new_y = one_step_ahead(y, lag, dt, alpha, trends, stride)

        # If the model returns NAN, the model was not able to be built, and an error of infinity should be returned
        if np.isnan(new_y):
            return np.inf
        
        # Write the calculated values
        y[len(y)] = new_y
        predict_days[len(predict_days)] = curr_date

        # Increment the date being predicted
        curr_date = curr_date + timedelta(days=1)

    # Find the predicted values
    y_pred = y[len(train):]

    # Calculate the error between the predicted values and the actual time series (if required)
    mse = np.inf
    if check_error:
        y_true = test['AVAILABLE_BIKE_STANDS']
        mse = mean_squared_error(y_true, y_pred)
    
    # If we are modelling post pandemic, we also need to extend the data assuming the pandemic happened
    if model_post_pandemic:
        # Find out how many days to predict
        curr_date = test['TIME'].iloc[-1].date() + timedelta(days=1)
        days_to_predict = (pd.to_datetime('2030-01-01').date()-curr_date).days

        # Initialise training data and location for storing values
        y_pandemic = pd.concat([train['AVAILABLE_BIKE_STANDS'], test['AVAILABLE_BIKE_STANDS']])
        pandemic_dates = pd.Series()

        # Predict all days required
        for _ in range(days_to_predict):
            new_y = one_step_ahead(y_pandemic, lag, dt, alpha, trends, stride)
            y_pandemic[len(y_pandemic)] = new_y
            pandemic_dates[len(pandemic_dates)] = curr_date
            curr_date = curr_date + timedelta(days=1)

        # Find the predicted values
        y_pandemic_pred = y_pandemic[(len(train) + len(test)):]
    
    if plot:
        # Split the test dataset into pandemic and post pandemic
        test = test.set_index(test['TIME'])
        pandemic = test[:'2022-01-28']
        post_pandemic = test['2022-01-28':]

        # Plot actual data from train and test datasets
        plt.plot(train['TIME'], train["AVAILABLE_BIKE_STANDS"], color='blue', label="Actual pre-pandemic bike usage")
        if post_pandemic.shape[0] == 0:
            plt.plot(test['TIME'], test["AVAILABLE_BIKE_STANDS"], color='green', label="Actual pandemic bike usage")
        else:
            plt.plot(pandemic['TIME'], pandemic["AVAILABLE_BIKE_STANDS"], color='green', label="Actual pandemic bike usage")
            plt.plot(post_pandemic['TIME'], post_pandemic["AVAILABLE_BIKE_STANDS"], color='red', label="Actual post-pandemic bike usage")

        # Plot predicted data 
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

# Cross validate to find the best hyperparameters
def cross_validation_on_pre_pandemic(pre_pandemic: pd.DataFrame, dt: float):
    # Set up potential hyperparameter values
    c_vals = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    lags = np.arange(start=1, stop=30)
    trends = ["short-term", "week", "month", "half-year"]
    strides = np.arange(start=1, stop=10)
    trend_combinations = [trends[i:j] for i in range(len(trends)) for j in range(i + 1, len(trends) + 1)]

    # Set up initial cross validation values
    current_min_error = np.inf
    current_min_c_val, current_min_lag, current_min_stride = 0, 0, 0
    current_min_trends = []

    # Split dataset into train/test
    train_size = math.floor(0.8*(pre_pandemic.shape[0]))
    train, test = pre_pandemic[0:train_size], pre_pandemic[train_size:]

    # Loop through all possible hyperparameter values and find minimum error
    for c in tqdm(c_vals, desc="Cross-validation"):
        for lag in lags:
            for trend_combination in trend_combinations:
                for stride in strides:
                    # Calculate alpha value given C
                    alpha = (1/(2*c)) 

                    # Check error with hyperparameters
                    err = multi_step_prediction(train=train, test=test, lag=lag, dt=dt, alpha=alpha, trends=trend_combination, stride=stride, plot=False, check_error=True, model_post_pandemic=False)
                    
                    # If the error has decreased, update hyperparameters
                    if err < current_min_error:
                        #print(f"{c}, {lag}, {stride},  {trend_combination}, {err}")
                        current_min_error = err
                        current_min_c_val = c
                        current_min_lag = lag
                        current_min_trends = trend_combination
                        current_min_stride = stride

    # Output and graph the best set of hyperparameters
    print(f"Lowest mean squared error with c={current_min_c_val}, lag={current_min_lag}, stride={current_min_stride}, and tracking trends {current_min_trends}. Error is {current_min_error}.")
    multi_step_prediction(train=train, test=test, lag=current_min_lag, dt=dt, alpha=(1/(2*current_min_c_val)), trends=current_min_trends, stride=current_min_stride, plot=True, check_error=False, model_post_pandemic=False)

# Add 1 year to a datetime
def add_one_year(input_datetime: datetime) -> datetime:
    return input_datetime + relativedelta(years=1)

# Make predictions by copying pre-pandemic data
def statistical_prediction(pre_pandemic_df: pd.DataFrame, pandemic_df: pd.DataFrame, post_pandemic_df: pd.DataFrame, include_post_pandemic: bool, date_to_predict_to: str):
    # Find the dataset for the year of 2019
    start_year_index = pre_pandemic_df[pre_pandemic_df['TIME'].dt.date == pd.to_datetime('2019-01-01').date()].index[0]
    end_year_index = pre_pandemic_df[pre_pandemic_df['TIME'].dt.date == pd.to_datetime('2020-01-01').date()].index[0]
    year_df = pre_pandemic_df[start_year_index:end_year_index].copy().reset_index(drop=True)
    year = year_df['AVAILABLE_BIKE_STANDS']
    time = year_df['TIME']

    # Calculate how many days need to be predicted
    days_to_predict = (pd.to_datetime(date_to_predict_to).date()-pd.to_datetime('2020-01-01').date()).days

    # Set up location for predictions
    usage = pd.Series()
    date_series = pd.Series()

    # Make predictions for all required full years
    for _ in range(math.floor(days_to_predict/365)):
        usage = pd.concat([usage, year])
        date_series = pd.concat([date_series, (time.apply(add_one_year))])
        time = time.apply(add_one_year)

    # Make predictions for all required remaining days
    for index in range(days_to_predict % 365):
        usage[len(usage)] = year[index]
        date_series[len(date_series)] = add_one_year(time[index])

    # Plot given data
    plt.plot(pre_pandemic_df['TIME'], pre_pandemic_df['AVAILABLE_BIKE_STANDS'], color='blue', label="Actual pre-pandemic bike usage")
    plt.plot(pandemic_df['TIME'], pandemic_df['AVAILABLE_BIKE_STANDS'], color='green', label="Actual pandemic bike usage")
    if include_post_pandemic:
        plt.plot(post_pandemic_df['TIME'], post_pandemic_df['AVAILABLE_BIKE_STANDS'], color='red', label="Actual post-pandemic bike usage")

    # Plot predicted bike usage 
    plt.plot(date_series, usage, color='lime', label="Predicted bike usage without pandemic")
    plt.xlabel("Time")
    plt.ylabel("Bike usage")
    plt.suptitle("Predicted bike usage given no pandemic")
    plt.title(f"using statistical prediction (copying pattern from 2019)", fontsize=10)
    plt.legend()
    plt.show()

# Read in the daily average bike usages
def read_daily_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    pre_pandemic = pd.read_csv("data/daily_pre-pandemic.csv", parse_dates=['TIME'])
    pandemic = pd.read_csv("data/daily_pandemic.csv", parse_dates=['TIME'])
    post_pandemic = pd.read_csv("data/daily_post-pandemic.csv", parse_dates=['TIME'])
    return pre_pandemic, pandemic, post_pandemic

def main():
    # Read in data
    pre_pandemic, pandemic, post_pandemic = read_daily_data()

    # Calculate the gap between datapoints
    t_full=pd.array(pd.DatetimeIndex(pre_pandemic.iloc[:,1]).astype(np.int64))/1000000000
    dt = t_full[1]-t_full[0]

    #cross_validation_on_pre_pandemic(pre_pandemic, dt)

    # Predict for pandemic period
    multi_step_prediction(train=pre_pandemic, test=pandemic, lag=14, dt=dt, alpha=1/(2*0.0001), trends=["short-term", "week", "month"], stride=3, plot=True, check_error=False, model_post_pandemic=False)
    multi_step_prediction(train=pre_pandemic, test=pandemic, lag=1, dt=dt, alpha=1/(2*0.0001), trends=["short-term", "week", "month", "half-year"], stride=3, plot=True, check_error=False, model_post_pandemic=False)
    statistical_prediction(pre_pandemic_df=pre_pandemic, pandemic_df=pandemic, post_pandemic_df=None, include_post_pandemic= False, date_to_predict_to=pandemic['TIME'].iloc[-1])
    
    # Predict for post-pandemic period
    multi_step_prediction(train=pre_pandemic, test=pd.concat([pandemic, post_pandemic]), lag=14, dt=dt, alpha=1/(2*0.0001), trends=["short-term", "week", "month"], stride=3, plot=True, check_error=False, model_post_pandemic=True)
    multi_step_prediction(train=pre_pandemic, test=pd.concat([pandemic, post_pandemic]), lag=1, dt=dt, alpha=1/(2*0.0001), trends=["short-term", "week", "month", "half-year"], stride=3, plot=True, check_error=False, model_post_pandemic=True)
    statistical_prediction(pre_pandemic_df=pre_pandemic, pandemic_df=pandemic, post_pandemic_df=post_pandemic, include_post_pandemic=True, date_to_predict_to='2030-01-01')

if __name__ == "__main__":
    main()