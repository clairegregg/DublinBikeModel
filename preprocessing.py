import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

COLUMNS_OLD = ["TIME","BIKE STANDS","AVAILABLE BIKE STANDS","AVAILABLE BIKES"]
COLUMNS_NEW = ["TIME","BIKE_STANDS","AVAILABLE_BIKE_STANDS","AVAILABLE_BIKES"]

# Split a file into 2 dataframes along the split_regex
def split_df(filename: str, split_regex: str) -> (pd.DataFrame, pd.DataFrame):
    if "Q" in filename:
        df = pd.read_csv(filename, usecols=COLUMNS_OLD)
        df.rename(columns={"BIKE STANDS": "BIKE_STANDS", 'AVAILABLE BIKE STANDS': 'AVAILABLE_BIKE_STANDS', 'AVAILABLE BIKES': 'AVAILABLE_BIKES'}, inplace=True)
    else:
        df = pd.read_csv(filename, usecols=COLUMNS_NEW)

    index = df[df["TIME"].str.contains(split_regex)].index[0]
    return df[:index], df[index:]

# Read all data from a folder and return it in a dataframe
def read_data(folder: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    dfs = {}

    # Split March 2020 into pre and post pandemic
    # March 13th, the first day schools were closed (https://www.irishtimes.com/health/2023/05/05/covid-emergency-is-over-20-key-moments-of-pandemic-that-changed-the-world/)
    pre_pandemic_mar, pandemic_mar = split_df(f"{folder}/start-pandemic/2020Q1.csv","2020-03-17 .*")

    # Split January 2022 into pre and post pandemic
    # Jan 28th 2022, the day the HSE stopped releasing COVID-19 figures (https://www.irishtimes.com/health/2023/05/05/covid-emergency-is-over-20-key-moments-of-pandemic-that-changed-the-world/)
    pandemic_jan, post_pandemic_jan = split_df(f"{folder}/end-pandemic/2022January.csv", "2022-01-28 .*")

    for time_period in ["pre-pandemic", "pandemic", "post-pandemic"]:
        period_df = []
        for subdir, _, files in os.walk(f"{folder}/{time_period}"):
            for file in tqdm(files, desc=f"Reading {subdir}"):
                if "Q" in file:
                    df = pd.read_csv(os.path.join(subdir, file), usecols=COLUMNS_OLD)
                    df.rename(columns={"BIKE STANDS": "BIKE_STANDS", 'AVAILABLE BIKE STANDS': 'AVAILABLE_BIKE_STANDS', 'AVAILABLE BIKES': 'AVAILABLE_BIKES'}, inplace=True)
                else:
                    df = pd.read_csv(os.path.join(subdir, file), usecols=COLUMNS_NEW)
                period_df.append(df)
                del df
        
            match time_period:
                case "pre-pandemic":
                    period_df.append(pre_pandemic_mar)
                    del pre_pandemic_mar
                case "pandemic":
                    period_df.insert(0,pandemic_mar)
                    period_df.append(pandemic_jan)
                    del pandemic_mar
                    del pandemic_jan
                case "post-pandemic":
                    period_df.insert(0,post_pandemic_jan)
                    del post_pandemic_jan
            
        dfs[time_period] = pd.concat(period_df, ignore_index=True)
        
    return dfs["pre-pandemic"], dfs["pandemic"], dfs["post-pandemic"]

# Round a time string to the nearest 5 minutes in a datetime object
def round_to_5_minutes(time: str) -> datetime:
    # Parse the input string into a datetime object
    dt_format = "%Y-%m-%d %H:%M:%S"
    dt = datetime.strptime(time, dt_format)

    # Round down to the nearest 5 minutes
    rounded_dt = datetime(dt.year, dt.month, dt.day, dt.hour, (dt.minute // 5) * 5)

    return rounded_dt

# Combine time/dates of different stations into one value per time
def combine_dates(df: pd.DataFrame, period: str) -> pd.DataFrame:
    # Round times to nearest 5 minutes
    tqdm.pandas(desc=f"Rounding {period} times to closest 5 minutes.")
    df['TIME'] = df['TIME'].progress_apply(round_to_5_minutes)

    # Aggregate times together, with all counts summed
    return df.groupby(df['TIME'], as_index=False).aggregate({'BIKE_STANDS': 'sum', 'AVAILABLE_BIKE_STANDS': 'sum', 'AVAILABLE_BIKES': 'sum'})

def plot_period(df: pd.DataFrame, period: str):
    plt.plot(df['TIME'], df['AVAILABLE_BIKE_STANDS'])
    plt.title(f"Available bikes in the {period} period")
    plt.xlabel("Time/Date")
    plt.ylabel("Number of available bikes")
    plt.show()

def main():
    pre_pandemic, pandemic, post_pandemic = read_data("data")
    pre_pandemic, pandemic, post_pandemic = combine_dates(pre_pandemic, "pre-pandemic"), combine_dates(pandemic, "pandemic"), combine_dates(post_pandemic, "post-pandemic")
    plot_period(pre_pandemic, "pre-pandemic")

if __name__ == "__main__":
    main()