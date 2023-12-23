import os
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
from tqdm import tqdm

# Read all data from a folder and return it in a dataframe
def read_data(folder):
    dfs = []

    for subdir, _, files in os.walk(folder):
        for file in tqdm(files, desc=f"Reading {folder}"):
            if file.endswith(".csv"):
                path = os.path.join(subdir, file)
                df = pd.read_csv(path)
                dfs.append(df)
                del df

    return pd.concat(dfs, ignore_index=True)

def main():
    pre_pandemic_data = read_data("data/pre-pandemic")
    
    print("Pre-Pandemic Data:")
    print(pre_pandemic_data.head())

if __name__ == "__main__":
    main()