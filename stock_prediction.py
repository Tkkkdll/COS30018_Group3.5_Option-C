# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)
# Modified: Task C.2 - Added load_data() function with multiple features,
#           NaN handling, configurable train/test split, local caching, and per-column scalers

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install yfinance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import tensorflow as tf
import yfinance as yf
import os
import random

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer

# Set seeds for reproducibility so we get the same results after rerunning several times
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------
COMPANY = 'CBA.AX'

TRAIN_START = '2020-01-01'     # Start date to read
TRAIN_END = '2024-07-02'       # End date to read

PRICE_VALUE = "adjclose"       # Feature to predict (changed from "Close" to use adjclose)

# Number of days to look back to base the prediction
PREDICTION_DAYS = 60 # Original

LOOKUP_STEP = 1        # How many days ahead to predict (1 = next day)
TEST_SIZE = 0.2        # 20% of data used for testing
SPLIT_BY_DATE = True   # Split train/test chronologically (True) or randomly (False)
FEATURE_COLUMNS = ['adjclose', 'volume', 'open', 'high', 'low', 'close']

#------------------------------------------------------------------------------
# Helper function for shuffling
#------------------------------------------------------------------------------
def shuffle_in_unison(a, b):
    """
    Shuffles two arrays (a and b) in the same order.
    This is needed because we want X and y to remain aligned after shuffling —
    if we shuffled them independently, each input would no longer match its label.
    """
    state = np.random.get_state()   # Save current random state
    np.random.shuffle(a)            # Shuffle array a
    np.random.set_state(state)      # Restore the same random state
    np.random.shuffle(b)            # Shuffle b in the same order as a

#------------------------------------------------------------------------------
# Load Data
## TO DO:
# 1) Check if data has been saved before. 
# If so, load the saved data
# If not, save the data into a directory
#------------------------------------------------------------------------------
def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1,
              split_by_date=True, test_size=0.2,
              feature_columns=['adjclose', 'volume', 'open', 'high', 'low', 'close'],
              start_date="2010-01-01", end_date=None,
              data_dir="data"):
    """
    Downloads (or loads from local cache) historical stock data, then scales,
    shuffles, and splits it into training and testing sets.

    Parameters:
        ticker          (str):   Stock ticker symbol e.g. 'CBA.AX'
        n_steps         (int):   Window size — how many past days to use per prediction
        scale           (bool):  Whether to normalise feature columns to [0, 1]
        shuffle         (bool):  Whether to shuffle training and testing sets
        lookup_step     (int):   How many days ahead to predict (1 = next day)
        split_by_date   (bool):  True = chronological split; False = random split
        test_size       (float): Proportion of data reserved for testing e.g. 0.2 = 20%
        feature_columns (list):  List of column names to use as model input features
        start_date      (str):   Start date for data download, format 'YYYY-MM-DD'
        end_date        (str):   End date for data download; None defaults to today
        data_dir        (str):   Directory to save/load cached CSV files

    Returns:
        result (dict) containing:
            'df'            — original dataframe
            'column_scaler' — dict of MinMaxScaler per feature column
            'last_sequence' — most recent sequence for predicting beyond dataset
            'X_train'       — training input sequences
            'y_train'       — training labels
            'X_test'        — testing input sequences
            'y_test'        — testing labels
            'test_df'       — subset of original df for the test period
    """

    # Create the data directory if it doesn't already exist
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    # Build a unique filename based on ticker and date range
    # so each combination gets its own cached file
    end_str = end_date if end_date else "today"
    cache_filename = os.path.join(data_dir, f"{ticker}_{start_date}_{end_str}.csv")

    if os.path.isfile(cache_filename):
        # Local copy found — load it to avoid re-downloading and hitting rate limits
        print(f"Loading cached data from: {cache_filename}")
        df = pd.read_csv(cache_filename, index_col=0, parse_dates=True)
    else:
        # No local copy — download from Yahoo Finance
        print(f"Downloading {ticker} data from {start_date} to {end_str}...")
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        # yfinance returns MultiIndex columns like ('Adj Close', 'CBA.AX')
        # We flatten to single level and lowercase: 'adjclose', 'open', etc.
        df.columns = [col[0].lower().replace(' ', '') for col in df.columns]
        # Save to CSV for future use — next run will load from cache instead
        df.to_csv(cache_filename)
        print(f"Data saved to: {cache_filename}")

    # Handle NaN (missing) values
    # Stock data can have NaN values due to public holidays or data gaps.
    # Forward fill uses the last known value; backward fill handles NaNs at the start.
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()

    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

    # add date as a column
    # The dataframe index is already the date, but we add it as a regular column
    # so it can travel through the sequence pipeline and be retrieved later
    if "date" not in df.columns:
        df["date"] = df.index

    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            # np.expand_dims adds an extra dimension: shape (n,) → (n, 1)
            # This is required because MinMaxScaler expects a 2D array as input
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            # Store each scaler individually so we can inverse_transform later
            # e.g. column_scaler['adjclose'].inverse_transform(predicted_values)
            column_scaler[column] = scaler
        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

    # add the target column (label) by shifting by `lookup_step`
    # shift(-lookup_step) moves values up: df['future'][i] = df['adjclose'][i + lookup_step]
    # This means each row's label is the price 'lookup_step' days in the future
    df['future'] = df['adjclose'].shift(-lookup_step)

    # last `lookup_step` columns contains NaN in future column
    # get them before dropping NaNs
    # These rows will be used to predict prices beyond the dataset
    last_sequence = np.array(df[feature_columns].tail(lookup_step))

    # drop NaNs
    df.dropna(inplace=True)

    sequence_data = []
    # deque(maxlen=n_steps) is like a fixed-size queue — when full, adding a new
    # item automatically removes the oldest one, making it perfect for a sliding window
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        # Only start saving sequences once we have a full window of n_steps rows
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence

    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        # This is more realistic for time-series — the model trains on past and tests on future
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"]  = X[train_samples:]
        result["y_test"]  = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:
        # split the dataset randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                test_size=test_size, shuffle=shuffle)

    # get the list of test set dates
    # X_test has shape (samples, n_steps, n_features + 1)
    # [:, -1, -1] gets the last timestep of each sequence, then the date value from that row
    dates = result["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    # [:, :, :len(feature_columns)] keeps all timesteps but drops the last date column
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

    return result

#------------------------------------------------------------------------------
# Load data using the new load_data() function
#------------------------------------------------------------------------------
data = load_data(
    ticker=COMPANY,
    n_steps=PREDICTION_DAYS,
    scale=True,
    shuffle=False,  # Don't shuffle for time-series data
    lookup_step=LOOKUP_STEP,
    split_by_date=SPLIT_BY_DATE,
    test_size=TEST_SIZE,
    feature_columns=FEATURE_COLUMNS,
    start_date=TRAIN_START,
    end_date=TRAIN_END
)

x_train = data["X_train"]
y_train = data["y_train"]

#------------------------------------------------------------------------------
# Prepare Data
## To do:
# 1) Check if data has been prepared before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Use a different price value eg. mid-point of Open & Close
# 3) Change the Prediction days
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
#------------------------------------------------------------------------------
model = Sequential() # Basic neural network
# See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# for some useful examples

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
# This is our first hidden layer which also specifies an input layer.
# That's why we specify the input shape for this layer;
# i.e. the format of each training example
# Note: input_shape now uses x_train.shape[2] for number of features
# (previously hardcoded to 1 since only Close price was used)
# For some advanced explanation of return_sequences:
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
# As explained there, for a stacked LSTM, you must set return_sequences=True 
# when stacking LSTM layers so that the next LSTM layer has a 
# three-dimensional sequence input. 

# Finally, units specifies the number of nodes in this layer.
# This is one of the parameters you want to play with to see what number
# of units will give you better prediction quality (for your problem)

model.add(Dropout(0.2))
# The Dropout layer randomly sets input units to 0 with a frequency of 
# rate (= 0.2 above) at each step during training time, which helps 
# prevent overfitting (one of the major problems of ML). 

model.add(LSTM(units=50, return_sequences=True))
# More on Stacked LSTM:
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1)) 
# Prediction of the next closing value of the stock price

# We compile the model by specify the parameters for the model
# See lecture Week 6 (COS30018)
model.compile(optimizer='adam', loss='mean_squared_error')
# The optimizer and loss are two important parameters when building an 
# ANN model. Choosing a different optimizer/loss can affect the prediction
# quality significantly. You should try other settings to learn; e.g.
    
# optimizer='rmsprop'/'sgd'/'adadelta'/...
# loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

# Now we are going to train this model with our training data 
# (x_train, y_train)
model.fit(x_train, y_train, epochs=25, batch_size=32)
# Other parameters to consider: How many rounds(epochs) are we going to 
# train our model? Typically, the more the better, but be careful about
# overfitting!
# What about batch_size? Well, again, please refer to 
# Lecture Week 6 (COS30018): If you update your model for each and every 
# input sample, then there are potentially 2 issues: 1. If you training 
# data is very big (billions of input samples) then it will take VERY long;
# 2. Each and every input can immediately makes changes to your model
# (a souce of overfitting). Thus, we do this in batches: We'll look at
# the aggreated errors/losses from a batch of, say, 32 input samples
# and update our model based on this aggregated loss.

# TO DO:
# Save the model and reload it
# Sometimes, it takes a lot of effort to train your model (again, look at
# a training data with billions of input samples). Thus, after spending so 
# much computing power to train your model, you may want to save it so that
# in the future, when you want to make the prediction, you only need to load
# your pre-trained model and run it on the new input for which the prediction
# need to be made.

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------
x_test = data["X_test"]
y_test = data["y_test"]

predicted_prices = model.predict(x_test)
predicted_prices = data["column_scaler"][PRICE_VALUE].inverse_transform(predicted_prices)
# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation 

actual_prices = data["column_scaler"][PRICE_VALUE].inverse_transform(y_test.reshape(-1, 1))

#------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
#------------------------------------------------------------------------------

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------
real_data = data["last_sequence"][-PREDICTION_DAYS:]
# np.expand_dims adds a batch dimension: (n_steps, n_features) -> (1, n_steps, n_features)
real_data = np.expand_dims(real_data, axis=0)

prediction = model.predict(real_data)
prediction = data["column_scaler"][PRICE_VALUE].inverse_transform(prediction)
print(f"Prediction: {prediction}")

# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day 
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem 
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??