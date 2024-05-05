# Qian Mao

# In[1]:

from this import d
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


LABEL2DIG = {'chat': 0, 'voip': 1, 'trap2p': 2, 'stream': 3, 'file_trans': 4, 'email': 5,
             'vpn_chat': 6, 'vpn_voip': 7, 'vpn_trap2p': 8, 'vpn_stream': 9, 'vpn_file_trans': 10, 'vpn_email': 11}
DIG2LABEL = {v: k for k, v in LABEL2DIG.items()}
num_classes = 12

# random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
# X = np.array([[1, 11], [2, 22], [3,33], [4, 44]])
# y = np.array([111,222, 333, 444])
# X, y = shuffle(X, y)
# print(X, y)

def DNN_scheme1():
    X_train = np.load('../data/X_train.npy', allow_pickle=True)
    y_train = np.load('../data/y_train.npy', allow_pickle=True)
    # shuffle the date
    #X_train, y_train = shuffle(X_train, y_train)

    X_train = X_train.astype('float32')

    print('X_train dimension:', np.shape(X_train))
    print('y_train dimension:', np.shape(y_train))
    #print(y_train[3000:4000])
    
    clw = []
    nums = np.zeros(6)
    maxsize = 0
    print('-'*20)
    for cat in np.unique(y_train):
        size = np.shape(np.where(y_train == cat))[1]
        #nums[LABEL2DIG[cat]] = size
        print(DIG2LABEL[cat]+": "+str(size))
        clw.append(1/size)
        if(size > maxsize):
            maxsize = size
    print('-'*20)

    clw = [i*maxsize for i in clw]

    y_train = to_categorical(y_train, num_classes = num_classes)
    #print(y_train[:100])

    X_train = normalize(X_train, norm='max', axis=0, copy=True, return_norm=False)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    #ds_train = tf.data.Dataset.from_tensor_slices(X_train, y_train)
    #ds_test = tf.data.Dataset.from_tensor_slices((dict(X_test), y_test))
    
    print(np.shape(X_train))
    print(np.shape(y_train))
    print(np.shape(X_test))
    print(np.shape(y_test))

    input_dimension = np.shape(X_train)[1]
    print(input_dimension)
    #print("show an input sample: ", X_train[1])

    # DNN model
    # input
    input = keras.Input(shape = (input_dimension))
    # hidden layer 1
    x = layers.Dense(120, activation="relu")(input)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Dropout(0.5)(x)
    # hidden layer 2
    x = layers.Dense(120, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Dropout(0.25)(x)
    # hidden layer 3
    x = layers.Dense(50, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Dropout(0.2)(x)
    # hidden layer 4
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Dropout(0.2)(x)
    
    output = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(input, output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy", keras.metrics.Precision()])
    callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)
    model.fit(X_train, y_train, batch_size=64, epochs=100, validation_split=0.2, callbacks=[callback])
    #model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

    model.save("../savedModels/my_model_DNN_scheme1")

    results = model.evaluate(X_test, y_test, batch_size=128)
    print("test loss, test accuracy, test precision:", results)


def DNN_scheme2():
    X_train = np.load('../data/X_train.npy', allow_pickle=True)
    y_train = np.load('../data/y_train.npy', allow_pickle=True)
    # shuffle the date
    #X_train, y_train = shuffle(X_train, y_train)

    X_train = X_train.astype('float32')

    print('X_train dimension:', np.shape(X_train))
    print('y_train dimension:', np.shape(y_train))
    #print(y_train[3000:4000])
    
    clw = []
    nums = np.zeros(6)
    maxsize = 0
    print('-'*20)
    for cat in np.unique(y_train):
        size = np.shape(np.where(y_train == cat))[1]
        #nums[LABEL2DIG[cat]] = size
        print(DIG2LABEL[cat]+": "+str(size))
        clw.append(1/size)
        if(size > maxsize):
            maxsize = size
    print('-'*20)

    clw = [i*maxsize for i in clw]

    y_train = to_categorical(y_train, num_classes = num_classes)
    #print(y_train[:100])

    X_train = normalize(X_train, norm='max', axis=0, copy=True, return_norm=False)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    #ds_train = tf.data.Dataset.from_tensor_slices(X_train, y_train)
    #ds_test = tf.data.Dataset.from_tensor_slices((dict(X_test), y_test))
    
    print(np.shape(X_train))
    print(np.shape(y_train))
    print(np.shape(X_test))
    print(np.shape(y_test))

    input_dimension = np.shape(X_train)[1]
    print(input_dimension)
    #print("show an input sample: ", X_train[1])
    
    # DNN model
    # input
    input = keras.Input(shape = (input_dimension))
    # hidden layer 1
    x = layers.Dense(1200, activation="relu")(input)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Dropout(0.5)(x)
    # hidden layer 2
    x = layers.Dense(980, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Dropout(0.5)(x)
    # hidden layer 4
    x = layers.Dense(640, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Dropout(0.25)(x)
    # hidden layer 5
    x = layers.Dense(460, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Dropout(0.25)(x)
    # hidden layer 6
    x = layers.Dense(220, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Dropout(0.2)(x)
    # hidden layer 7
    x = layers.Dense(80, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Dropout(0.2)(x)
    
    output = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(input, output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy", keras.metrics.Precision()])
    callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)
    model.fit(X_train, y_train, batch_size=64, epochs=100, validation_split=0.2, callbacks=[callback])
    #model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

    model.save("../savedModels/my_model_DNN_scheme2")

    results = model.evaluate(X_test, y_test, batch_size=128)
    print("test loss, test accuracy, test precision:", results)


def DNN_scheme3():
    X_train = np.load('../data/X_train.npy', allow_pickle=True)
    y_train = np.load('../data/y_train.npy', allow_pickle=True)
    # shuffle the date
    #X_train, y_train = shuffle(X_train, y_train)

    X_train = X_train.astype('float32')

    print('X_train dimension:', np.shape(X_train))
    print('y_train dimension:', np.shape(y_train))
    #print(y_train[3000:4000])
    
    clw = []
    nums = np.zeros(6)
    maxsize = 0
    print('-'*20)
    for cat in np.unique(y_train):
        size = np.shape(np.where(y_train == cat))[1]
        #nums[LABEL2DIG[cat]] = size
        print(DIG2LABEL[cat]+": "+str(size))
        clw.append(1/size)
        if(size > maxsize):
            maxsize = size
    print('-'*20)

    clw = [i*maxsize for i in clw]

    y_train = to_categorical(y_train, num_classes = num_classes)
    #print(y_train[:100])

    X_train = normalize(X_train, norm='max', axis=0, copy=True, return_norm=False)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    #ds_train = tf.data.Dataset.from_tensor_slices(X_train, y_train)
    #ds_test = tf.data.Dataset.from_tensor_slices((dict(X_test), y_test))
    
    print(np.shape(X_train))
    print(np.shape(y_train))
    print(np.shape(X_test))
    print(np.shape(y_test))

    input_dimension = np.shape(X_train)[1]
    print(input_dimension)
    #print("show an input sample: ", X_train[1])
    
    ###########################################
    # try different DNN models
    # # DNN model 3-3 (3 hidden layers)
    # # input
    # input = keras.Input(shape = (input_dimension))
    # # hidden layer 1
    # x = layers.Dense(1600, activation="relu")(input)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.5)(x)
    # # hidden layer 2
    # x = layers.Dense(1200, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.25)(x)
    # # hidden layer 3
    # x = layers.Dense(220, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.2)(x)
    
    # output = layers.Dense(num_classes, activation="softmax")(x)
    # # end of model 3-3

    # # DNN model 3-4 (4 hidden layers)
    # # input
    # input = keras.Input(shape = (input_dimension))
    # # hidden layer 1
    # x = layers.Dense(1600, activation="relu")(input)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.5)(x)
    # # hidden layer 2
    # x = layers.Dense(1200, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.25)(x)
    # # hidden layer 3
    # x = layers.Dense(640, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.25)(x)
    # # hidden layer 4
    # x = layers.Dense(120, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.2)(x)
    
    # output = layers.Dense(num_classes, activation="softmax")(x)
    # # end of model 3-4

    # # DNN model 3-5 (5 hidden layers)
    # # input
    # input = keras.Input(shape = (input_dimension))
    # # hidden layer 1
    # x = layers.Dense(1600, activation="relu")(input)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.5)(x)
    # # hidden layer 2
    # x = layers.Dense(1200, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.5)(x)
    # # hidden layer 3
    # x = layers.Dense(880, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.25)(x)
    # # hidden layer 4
    # x = layers.Dense(640, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.25)(x)
    # # hidden layer 5
    # x = layers.Dense(220, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.2)(x)
    
    # output = layers.Dense(num_classes, activation="softmax")(x)
    # # end of model 3-5

    # DNN model 3-6 (6 hidden layers)
    # input
    input = keras.Input(shape = (input_dimension))
    # hidden layer 1
    x = layers.Dense(1600, activation="relu")(input)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Dropout(0.5)(x)
    # hidden layer 2
    x = layers.Dense(1200, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Dropout(0.5)(x)
    # hidden layer 3
    x = layers.Dense(880, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Dropout(0.25)(x)
    # hidden layer 4
    x = layers.Dense(640, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Dropout(0.25)(x)
    # hidden layer 5
    x = layers.Dense(640, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Dropout(0.2)(x)
    # hidden layer 6
    x = layers.Dense(220, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Dropout(0.2)(x)
    
    output = layers.Dense(num_classes, activation="softmax")(x)
    # end of model 3-6

    # # DNN model 3-7 (7 hidden layers)
    # # input
    # input = keras.Input(shape = (input_dimension))
    # # hidden layer 1
    # x = layers.Dense(1600, activation="relu")(input)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.5)(x)
    # # hidden layer 2
    # x = layers.Dense(1200, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.5)(x)
    # # hidden layer 3
    # x = layers.Dense(880, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.25)(x)
    # # hidden layer 4
    # x = layers.Dense(640, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.25)(x)
    # # hidden layer 5
    # x = layers.Dense(360, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.25)(x)
    # # hidden layer 6
    # x = layers.Dense(220, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.2)(x)
    # # hidden layer 7
    # x = layers.Dense(80, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.2)(x)
    
    # output = layers.Dense(num_classes, activation="softmax")(x)
    # # end of model 3-7

    # # DNN model 3-8 (8 hidden layers)
    # # input
    # input = keras.Input(shape = (input_dimension))
    # # hidden layer 1
    # x = layers.Dense(1600, activation="relu")(input)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.5)(x)
    # # hidden layer 2
    # x = layers.Dense(1200, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.5)(x)
    # # hidden layer 3
    # x = layers.Dense(1200, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.25)(x)
    # # hidden layer 4
    # x = layers.Dense(880, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.25)(x)
    # # hidden layer 5
    # x = layers.Dense(640, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.25)(x)
    # # hidden layer 6
    # x = layers.Dense(360, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.25)(x)
    # # hidden layer 7
    # x = layers.Dense(220, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.2)(x)
    # # hidden layer 8
    # x = layers.Dense(80, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.2)(x)
    
    # output = layers.Dense(num_classes, activation="softmax")(x)
    # # end of model 3-8

    # # DNN model 3-9 (9 hidden layers)
    # # input
    # input = keras.Input(shape = (input_dimension))
    # # hidden layer 1
    # x = layers.Dense(1600, activation="relu")(input)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.5)(x)
    # # hidden layer 2
    # x = layers.Dense(1200, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.5)(x)
    # # hidden layer 3
    # x = layers.Dense(1200, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.5)(x)
    # # hidden layer 4
    # x = layers.Dense(880, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.25)(x)
    # # hidden layer 5
    # x = layers.Dense(640, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.25)(x)
    # # hidden layer 6
    # x = layers.Dense(640, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.25)(x)
    # # hidden layer 7
    # x = layers.Dense(360, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.25)(x)
    # # hidden layer 8
    # x = layers.Dense(220, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.2)(x)
    # # hidden layer 9
    # x = layers.Dense(80, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.2)(x)
    
    # output = layers.Dense(num_classes, activation="softmax")(x)
    # # end of model 3-9

    # # DNN model 3-10 (10 hidden layers)
    # # input
    # input = keras.Input(shape = (input_dimension))
    # # hidden layer 1
    # x = layers.Dense(1600, activation="relu")(input)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.5)(x)
    # # hidden layer 2
    # x = layers.Dense(1600, activation="relu")(input)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.5)(x)
    # # hidden layer 3
    # x = layers.Dense(1200, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.5)(x)
    # # hidden layer 4
    # x = layers.Dense(1200, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.25)(x)
    # # hidden layer 5
    # x = layers.Dense(880, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.25)(x)
    # # hidden layer 6
    # x = layers.Dense(640, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.25)(x)
    # # hidden layer 7
    # x = layers.Dense(640, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.25)(x)
    # # hidden layer 8
    # x = layers.Dense(360, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.25)(x)
    # # hidden layer 9
    # x = layers.Dense(220, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.2)(x)
    # # hidden layer 10
    # x = layers.Dense(80, activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Dropout(0.2)(x)
    
    # output = layers.Dense(num_classes, activation="softmax")(x)
    # # end of model 3-10
    # ##############################################################
    
    model = keras.Model(input, output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy", keras.metrics.Precision()])
    callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)
    model.fit(X_train, y_train, batch_size=64, epochs=100, validation_split=0.2, callbacks=[callback])
    #model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

    model.save("../savedModels/my_model_DNN_scheme3")

    results = model.evaluate(X_test, y_test, batch_size=128)
    print("test loss, test accuracy, test precision:", results)


def CNN_scheme1():
    X_train = np.load('../data/X_train.npy', allow_pickle=True)
    y_train = np.load('../data/y_train.npy', allow_pickle=True)
    # shuffle the date
    #X_train, y_train = shuffle(X_train, y_train)

    X_train = X_train.astype('float32')

    print('X_train dimension:', np.shape(X_train))
    print('y_train dimension:', np.shape(y_train))
    #print(y_train[3000:4000])
    
    clw = []
    nums = np.zeros(6)
    maxsize = 0
    print('-'*20)
    for cat in np.unique(y_train):
        size = np.shape(np.where(y_train == cat))[1]
        #nums[LABEL2DIG[cat]] = size
        print(DIG2LABEL[cat]+": "+str(size))
        clw.append(1/size)
        if(size > maxsize):
            maxsize = size
    print('-'*20)

    clw = [i*maxsize for i in clw]

    y_train = to_categorical(y_train, num_classes = num_classes)

    X_train = normalize(X_train, norm='max', axis=0, copy=True, return_norm=False)

    # pad columns for X_train and decide num_sensors
    period_len = 3
    input_dimension = np.shape(X_train)[1]
    if input_dimension % 3 == 2:
        X_train = np.insert(X_train, 0, 0, axis=1)
    elif input_dimension % 3 == 1:
        X_train = np.insert(X_train, 0, 0, axis=1)
        X_train = np.insert(X_train, 0, 0, axis=1)
    input_dimension = np.shape(X_train)[1]
    num_period = int(input_dimension/period_len)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    #ds_train = tf.data.Dataset.from_tensor_slices(X_train, y_train)
    #ds_test = tf.data.Dataset.from_tensor_slices((dict(X_test), y_test))
    
    print("X_train shape: ", np.shape(X_train))
    print("y_train shape: ", np.shape(y_train))
    print("X_test shape: ", np.shape(X_test))
    print("y_test shape: ", np.shape(y_test))

    #print("show an input sample: ", X_train[1])
    
    # 1-D CNN model  
    model = keras.Sequential()
    model.add(layers.Reshape((num_period, period_len), input_shape=(input_dimension,)))
    model.add(layers.Conv1D(100, 2, activation='relu', input_shape=(num_period, period_len)))
    #model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(100, 2, activation='relu'))    # 100 filters with 80 kernel size
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(580, activation="relu")) #580
    # model.add(layers.Dropout(0.25))
    model.add(layers.Dense(64, activation="relu")) #220
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(num_classes, activation='softmax'))
    print(model.summary())

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy", keras.metrics.Precision()])
    callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)
    model.fit(X_train, y_train, batch_size=64, epochs=100, validation_split=0.2, callbacks=[callback])
    #model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

    model.save("../savedModels/my_model_CNN_scheme_1")

    results = model.evaluate(X_test, y_test, batch_size=128)
    print("test loss, test accuracy, test precision:", results)



def CNN_scheme2():
    X_train = np.load('../data/X_train.npy', allow_pickle=True)
    y_train = np.load('../data/y_train.npy', allow_pickle=True)
    # shuffle the date
    #X_train, y_train = shuffle(X_train, y_train)

    X_train = X_train.astype('float32')

    print('X_train dimension:', np.shape(X_train))
    print('y_train dimension:', np.shape(y_train))
    #print(y_train[3000:4000])
    
    clw = []
    nums = np.zeros(6)
    maxsize = 0
    print('-'*20)
    for cat in np.unique(y_train):
        size = np.shape(np.where(y_train == cat))[1]
        #nums[LABEL2DIG[cat]] = size
        print(DIG2LABEL[cat]+": "+str(size))
        clw.append(1/size)
        if(size > maxsize):
            maxsize = size
    print('-'*20)

    clw = [i*maxsize for i in clw]

    y_train = to_categorical(y_train, num_classes = num_classes)

    X_train = normalize(X_train, norm='max', axis=0, copy=True, return_norm=False)

    # pad columns for X_train and decide num_sensors
    period_len = 3
    input_dimension = np.shape(X_train)[1]
    if input_dimension % 3 == 2:
        X_train = np.insert(X_train, 0, 0, axis=1)
    elif input_dimension % 3 == 1:
        X_train = np.insert(X_train, 0, 0, axis=1)
        X_train = np.insert(X_train, 0, 0, axis=1)
    input_dimension = np.shape(X_train)[1]
    num_period = int(input_dimension/period_len)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    #ds_train = tf.data.Dataset.from_tensor_slices(X_train, y_train)
    #ds_test = tf.data.Dataset.from_tensor_slices((dict(X_test), y_test))
    
    print("X_train shape: ", np.shape(X_train))
    print("y_train shape: ", np.shape(y_train))
    print("X_test shape: ", np.shape(X_test))
    print("y_test shape: ", np.shape(y_test))

    #print("show an input sample: ", X_train[1])
    
    # 1-D CNN model  
    model = keras.Sequential()
    model.add(layers.Reshape((num_period, period_len), input_shape=(input_dimension,)))
    model.add(layers.Conv1D(100, 100, activation='relu', input_shape=(num_period, period_len)))
    #model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(100, 80, activation='relu'))    # 100 filters with 80 kernel size
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(580, activation="relu")) #580
    # model.add(layers.Dropout(0.25))
    model.add(layers.Dense(220, activation="relu")) #220
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(num_classes, activation='softmax'))
    print(model.summary())

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy", keras.metrics.Precision()])
    callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)
    model.fit(X_train, y_train, batch_size=64, epochs=100, validation_split=0.2, callbacks=[callback])
    #model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

    model.save("../savedModels/my_model_CNN_scheme_1")

    results = model.evaluate(X_test, y_test, batch_size=128)
    print("test loss, test accuracy, test precision:", results)


def CNN_scheme3():
    X_train = np.load('../data/X_train.npy', allow_pickle=True)
    y_train = np.load('../data/y_train.npy', allow_pickle=True)
    # shuffle the date
    #X_train, y_train = shuffle(X_train, y_train)

    X_train = X_train.astype('float32')

    print('X_train dimension:', np.shape(X_train))
    print('y_train dimension:', np.shape(y_train))
    #print(y_train[3000:4000])
    
    clw = []
    nums = np.zeros(6)
    maxsize = 0
    print('-'*20)
    for cat in np.unique(y_train):
        size = np.shape(np.where(y_train == cat))[1]
        #nums[LABEL2DIG[cat]] = size
        print(DIG2LABEL[cat]+": "+str(size))
        clw.append(1/size)
        if(size > maxsize):
            maxsize = size
    print('-'*20)

    clw = [i*maxsize for i in clw]

    y_train = to_categorical(y_train, num_classes = num_classes)

    X_train = normalize(X_train, norm='max', axis=0, copy=True, return_norm=False)

    # pad columns for X_train and decide num_sensors
    period_len = 3
    input_dimension = np.shape(X_train)[1]
    if input_dimension % 3 == 2:
        X_train = np.insert(X_train, 0, 0, axis=1)
    elif input_dimension % 3 == 1:
        X_train = np.insert(X_train, 0, 0, axis=1)
        X_train = np.insert(X_train, 0, 0, axis=1)
    input_dimension = np.shape(X_train)[1]
    num_period = int(input_dimension/period_len)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    #ds_train = tf.data.Dataset.from_tensor_slices(X_train, y_train)
    #ds_test = tf.data.Dataset.from_tensor_slices((dict(X_test), y_test))
    
    print("X_train shape: ", np.shape(X_train))
    print("y_train shape: ", np.shape(y_train))
    print("X_test shape: ", np.shape(X_test))
    print("y_test shape: ", np.shape(y_test))

    #print("show an input sample: ", X_train[1])
    
    ###########################################################
    # try varous CNN models
    # # 1-D CNN model 3.1
    # model = keras.Sequential()
    # model.add(layers.Reshape((num_period, period_len), input_shape=(input_dimension,)))
    # model.add(layers.Conv1D(100, 100, activation='relu', input_shape=(num_period, period_len)))
    # model.add(layers.GlobalAveragePooling1D())
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(120, activation="relu")) #220
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(num_classes, activation='softmax'))
    # print(model.summary())
    # # end of CNN 3.1

    # 1-D CNN model 3.2
    model = keras.Sequential()
    model.add(layers.Reshape((num_period, period_len), input_shape=(input_dimension,)))
    model.add(layers.Conv1D(100, 100, activation='relu', input_shape=(num_period, period_len)))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1200, activation="relu")) #1200
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(580, activation="relu")) #580
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(120, activation="relu")) #220
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(num_classes, activation='softmax'))
    print(model.summary())
    # end of CNN 3.2

    # # 1-D CNN model 3.3
    # model = keras.Sequential()
    # model.add(layers.Reshape((num_period, period_len), input_shape=(input_dimension,)))
    # model.add(layers.Conv1D(100, 100, activation='relu', input_shape=(num_period, period_len)))
    # #model.add(layers.MaxPooling1D(2))
    # model.add(layers.Conv1D(100, 80, activation='relu'))    # 100 filters with 80 kernel size
    # model.add(layers.GlobalAveragePooling1D())
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(120, activation="relu")) #220
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(num_classes, activation='softmax'))
    # print(model.summary())
    # # end of CNN 3.3

    # # 1-D CNN model 3.4
    # model = keras.Sequential()
    # model.add(layers.Reshape((num_period, period_len), input_shape=(input_dimension,)))
    # model.add(layers.Conv1D(100, 100, activation='relu', input_shape=(num_period, period_len)))
    # #model.add(layers.MaxPooling1D(2))
    # model.add(layers.Conv1D(100, 80, activation='relu'))    # 100 filters with 80 kernel size
    # model.add(layers.GlobalAveragePooling1D())
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(580, activation="relu")) #580
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(120, activation="relu")) #220
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(num_classes, activation='softmax'))
    # print(model.summary())
    # # end of CNN 3.4

    # # 1-D CNN model 3.5
    # model = keras.Sequential()
    # model.add(layers.Reshape((num_period, period_len), input_shape=(input_dimension,)))
    # model.add(layers.Conv1D(100, 100, activation='relu', input_shape=(num_period, period_len)))
    # #model.add(layers.MaxPooling1D(2))
    # model.add(layers.Conv1D(100, 80, activation='relu'))    # 100 filters with 80 kernel size
    # model.add(layers.GlobalAveragePooling1D())
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(1200, activation="relu")) #1200
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(580, activation="relu")) #580
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(120, activation="relu")) #220
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(num_classes, activation='softmax'))
    # print(model.summary())
    # # end of CNN 3.5

    # # 1-D CNN model 3.6
    # model = keras.Sequential()
    # model.add(layers.Reshape((num_period, period_len), input_shape=(input_dimension,)))
    # model.add(layers.Conv1D(100, 100, activation='relu', input_shape=(num_period, period_len)))
    # # model.add(layers.MaxPooling1D(2))
    # model.add(layers.Conv1D(100, 80, activation='relu'))    # 100 filters with 80 kernel size
    # model.add(layers.GlobalAveragePooling1D())
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(1600, activation="relu")) #1200
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(1200, activation="relu")) #1200
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(580, activation="relu")) #580
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(120, activation="relu")) #220
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(num_classes, activation='softmax'))
    # print(model.summary())
    # # end of CNN 3.6

    # # 1-D CNN model 3.7
    # model = keras.Sequential()
    # model.add(layers.Reshape((num_period, period_len), input_shape=(input_dimension,)))
    # model.add(layers.Conv1D(100, 100, activation='relu', input_shape=(num_period, period_len)))
    # # model.add(layers.MaxPooling1D(2))
    # model.add(layers.Conv1D(100, 80, activation='relu'))    # 100 filters with 80 kernel size
    # model.add(layers.Conv1D(100, 80, activation='relu'))    # 100 filters with 80 kernel size
    # model.add(layers.GlobalAveragePooling1D())
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(1200, activation="relu")) #1200
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(580, activation="relu")) #580
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(120, activation="relu")) #220
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(num_classes, activation='softmax'))
    # print(model.summary())
    # # end of CNN 3.7
    # end of test different CNN models
    ############################################################

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy", keras.metrics.Precision()])
    callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)
    model.fit(X_train, y_train, batch_size=64, epochs=100, validation_split=0.2, callbacks=[callback])
    #model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

    model.save("../savedModels/my_model_CNN_scheme_1")

    results = model.evaluate(X_test, y_test, batch_size=128)
    print("test loss, test accuracy, test precision:", results)


if __name__ == '__main__':
    #DNN_scheme1()
    #DNN_scheme2()
    #DNN_scheme3()
    #CNN_scheme1()
    #CNN_scheme2()
    CNN_scheme3()
    
            