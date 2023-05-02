from discretize import discretizeToCsv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from tensorflow import keras
from keras.layers import Normalization
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Input

predahead = 36
# larger lookback values cause increasing validation
lookback = 36
# train - test split, value is train percentage
splitratio = .6


def joinsets():
    discrete = pd.read_csv("discrete.csv")
    sentiment = pd.read_csv("hourlysentiment.csv")
    sentiment['Time'] = sentiment['Time'].apply(
        lambda x: datetime.fromisoformat(x) - timedelta(0, 0, 0, 0, 1))
    discrete['Time'] = discrete['discretehour'].apply(
        lambda x: datetime.fromisoformat(x))
    discrete = discrete.drop("discretehour", axis=1)
    joined = discrete.set_index("Time").join(sentiment.set_index("Time"))
    return joined.dropna()


def cleanset(dataset, dropcolumns=[]):
    if len(dropcolumns) == 0:
        quit()
    dataset = dataset.dropna().reset_index()
    dataset = dataset.drop(dataset.columns[dropcolumns], axis=1)
    if (dataset.isna().any().any() == True):
        quit()
    return dataset


def scaler(joined):
    name = datetime.now().strftime("%Y%m%d-%H%M%S") + "shaved"
    sc = MinMaxScaler(feature_range=(0, 1))
    scaledset = sc.fit_transform(joined)
    scaler_filename = "trainscaler.save"
    joblib.dump(sc, scaler_filename + name)
    return scaledset


def create_dataset_ahead(dataset, look_back, look_forward, skip_ahead=0, pattern=1, pricecolindex=0):
    x, y = [], []
    for i in range(len(dataset)):
        a = dataset[i:(i+look_back), :]
        if (i >= len(dataset) - look_back - 1):
            break
        x.append(a)
        # last value is the column index for price, changes depending on spreadsheet
        y.append(dataset[i+1+skip_ahead:i +
                 look_forward+1:pattern, pricecolindex])
    #
    testchop = dataset[:-37]
    # todo: what does this do?
    if (x[len(x)-1][0][7] != testchop[len(testchop)-1][7]):
        # important, as our data method returns a slightly smaller dataframe than the dataset
        print("frames not aligned, will cause bias and future knowledge")
        quit()
    return np.array(x), np.array(y)


# here we shape our dataset, since once we define our neural net, it is very picky about what input it will take. In this case, we are feeding it three dimensional data
# Imagine a block of data, where our X and Y axis are our normal meanprice/volume/etc and time, respectivelly.
# Now we have a Z axis of
def shape(scaledset):
    train = np.array(scaledset[:int(np.round(len(scaledset)*splitratio))])
    test = np.array(scaledset[int(np.round(len(scaledset)*splitratio)):])

    xtrain, ytrain = create_dataset_ahead(
        train, lookback, predahead, skip_ahead=0, pattern=4)
    xtest, ytest = create_dataset_ahead(
        test,  lookback, predahead, skip_ahead=0, pattern=4)
    xtrain = np.reshape(xtrain, (xtrain.shape))
    xtest = np.reshape(xtest, (xtest.shape))

    return xtrain, xtest, ytrain, ytest


def trainpred():
    joined = joinsets()
    dataset = cleanset(joined, dropcolumns=[0])
    dataset = scaler(dataset)
    xtrain, xtest, ytrain, ytest = shape(dataset)
    print("hi")
    name = "ghfdgdf"
    log_dir = "logs/fit/" + name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

    model = Sequential()
    model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0,
              unroll=False, use_bias=True, input_shape=(xtrain.shape[1], xtrain.shape[2]), return_sequences=True))
    model.add(Dropout(0.2, seed=42))
    model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid',
              recurrent_dropout=0, unroll=False, use_bias=True, return_sequences=True))
    model.add(Dropout(0.2, seed=42))
    model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid',
              recurrent_dropout=0, unroll=False, use_bias=True))
    model.add(Dropout(0.2, seed=42))
    model.add(Dense(ytrain.shape[1]))
    # model.add(Dense(predahead))
    # VERY GOOD LEARNING RATE AT 0.00001 200 epochs in, 36 back 12 ahead, 0.002 loss quickly
    # 0.0015 60 epochs in, 36 back 36 ahead, very close loss/val at 200 epochs
    model.compile(loss='mean_squared_error',
                  optimizer=tf.optimizers.Adam(learning_rate=0.0001))
    history = model.fit(xtrain, ytrain, validation_data=[xtest, ytest], callbacks=[
                        tensorboard_callback], epochs=500, batch_size=256)
    model.save('finalmaybe' + name)



trainpred()


# todo: remove skipped hours
# todo: check if shaping is correct
# todo: unscaling verification
# todo: implement rest of prediction
