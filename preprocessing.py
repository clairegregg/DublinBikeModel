import os
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
from tqdm import tqdm

def split_df(filename: str, split_regex: str) -> (pd.DataFrame, pd.DataFrame):
    df = pd.read_csv(filename)
    index = df[df["TIME"].str.contains(split_regex)].index[0]
    print(index)
    return df[:index], df[index:]

# Read all data from a folder and return it in a dataframe
def read_data(folder: str) -> dict[str, pd.DataFrame]:
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
            for file in files:
                df = pd.read_csv(os.path.join(subdir, file))
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
        
    return dfs

def main():
    data = read_data("data")
    
    print("Pre-Pandemic Data:")
    print(data["pre-pandemic"].head())

if __name__ == "__main__":
    main()