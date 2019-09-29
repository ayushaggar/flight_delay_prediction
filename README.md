## Objective
Predicting flight delay

**Input** -
1) Flight id 
2) Date of flight and it returns the predicted delay. 

**Output** :
1) Predicted delay 
2) Error log

**Note**: Python code is pep8 compliant

## Tools use 
> Python 3

> Main Libraries Used -
1) numpy
2) pandas
3) scikit

## Installing and Running

> 
```sh
$ cd flight_delay
$ pip install -r requirements.txt
``` 

For Running Script
```sh
$ python model.py
```

## Various terminologies in approach are -

1) STD : Scheduled time of departure
2) ATD : Actual time of departure
3) Flight delay time: Flight delay means the difference of ATD - STD. Delay in departure only

## Various steps in approach are -

1) Join date with departure time
2) Manage outlier data
3) Did feature engineering to extract important features
4) Convert category values into numeric data by using one-hot encoding
5) Split data to train and test
6) Gridsearch to find parameters
7) Save model in pkl format to use later



