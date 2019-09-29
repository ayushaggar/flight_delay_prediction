from sys import argv

import numpy as np  # linear algebra
import pandas as pd  # data processing
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import datetime


def train_model(train_data):
    df = pd.read_csv(train_data)
    # join date with departure time
    df['STD'] = df['DATE'] + ' ' + df['STD']
    df['ATD'] = df['DATE'] + ' ' + df['ATD']

    # string to datetime format and extract month and day
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['MONTH'] = df['DATE'].dt.month
    df['DAY'] = df['DATE'].dt.day
    df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek

    # get delay in departure time in seconds
    df['STD'] = pd.to_datetime(df['STD'])
    df['ATD'] = pd.to_datetime(df['ATD'])
    df['DELAY'] = (df['ATD'] - df['STD']).dt.seconds

    #  drop departure data as DEPARTURE_DELAY is counted
    df = df.drop(["STD", "ATD", "DATE"], axis=1)

    df['DURATION'] = df['DURATION'] / np.timedelta64(1, 's')

    #  drop arrival data as ARRIVAL_DELAY not counted
    df = df.drop(["STA", "ATA", "STATUS"], axis=1)

    # missing_df = df.isnull().sum(axis=0).reset_index()
    # missing_df.columns = ['variable', 'missing values']
    # print (missing_df)
    # print (df['FROM'].nunique(dropna = True))
    # print (df['TO'].nunique(dropna = True))
    # print (df['AIRCRAFT'].nunique(dropna = True))
    # print (df['FLIGHT_ID'].nunique(dropna = True))
    # print (df['AIRLINE'].nunique(dropna = True))

    # manage the extreme delays more than 9 hours can be accident etc
    df = df[df['DELAY'] <= 32400]

    # convert category values into numeric data by using one-hot encoding
    cols_encode = ['FROM', 'TO', 'AIRCRAFT', 'FLIGHT_ID', 'AIRLINE']
    df_encode = pd.get_dummies(df[cols_encode], drop_first=True)
    df = pd.concat([df, df_encode], axis=1)
    df = df.drop(cols_encode, axis=1)
    # print (df_encode.shape)
    # print (df_encode.head())
    # print (df.min(axis = 0))
    # print (df.max(axis = 0))

    


def main():
    if len(argv) != 2:
        print("Please provide only 4 argument which should be the input file name and save file name")
        return
    train_data = argv[1]
    train_model(train_data)


if __name__ == "__main__":
    main()
else:
    print ("Executed when imported")
