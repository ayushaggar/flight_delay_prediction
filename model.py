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

    # split data to train and test
    train_set, test_set, y_train, y_test = train_test_split(
        df.drop(["DELAY"], axis=1), df["DELAY"], random_state=25, test_size=0.2)

    # Transforms features between 0 and 1
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_set)
    X_train = scaler.transform(train_set)
    X_test = scaler.transform(test_set)

    # print (X_train.shape)
    print("model start")

    # gridsearch to find parameters
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': [5, 7],
            'max_features': [5, 10],
            'n_estimators': [100],
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    grid_result = gsc.fit(X_train, y_train)
    best_params = grid_result.best_params_
    print(best_params)
    final_model = RandomForestRegressor(
        max_depth=best_params["max_depth"],
        n_estimators=best_params["n_estimators"],
        random_state=False,
        verbose=False)
    final_model.fit(X_train, y_train)
    final_model.score(X_test, y_test)
    y_pred = final_model.predict(X_test)
    print (y_pred)
    with open('flight_delay.pkl', 'wb') as fid:
        pickle.dump(final_model, fid)


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
