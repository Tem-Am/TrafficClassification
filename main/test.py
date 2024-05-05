# testing
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

LABEL2DIG = {'chat': 0, 'voip': 1, 'trap2p': 2, 'stream': 3, 'file_trans': 4, 'email': 5,
             'vpn_chat': 6, 'vpn_voip': 7, 'vpn_trap2p': 8, 'vpn_stream': 9, 'vpn_file_trans': 10, 'vpn_email': 11}
DIG2LABEL = {v: k for k, v in LABEL2DIG.items()}
num_classes = 12

def testing():
    X_train = np.load('../data/X_train.npy')
    X_test_origianl = np.load('../data/X_test.npy')
    y_test_origianl = np.load('../data/y_test.npy')
    X_test_origianl = X_test_origianl.astype('float32')
    #y_train = y_train
    print('X_test_origianl:', np.shape(X_test_origianl))
    print('y_test_origianl:', np.shape(y_test_origianl))

    y_test = to_categorical(y_test_origianl, num_classes = num_classes)
    
    # pad zeros to X_test
    X_train_demension = np.shape(X_train)[1]
    X_test = []
    for index in range(np.shape(X_test_origianl)[0]):
        X_test_origianl_demension = np.shape(X_test_origianl[index,])[0]
        print("sample ", index, "demension: ", X_test_origianl_demension)
        demension_diff = X_train_demension - X_test_origianl_demension
        print("demension difference: ", demension_diff)
        #print(X_test[index,])
        X_test.append(np.pad(X_test_origianl[index,], (0, demension_diff), 'constant'))
        #print(X_test[index,])

    X_test = normalize(X_test, norm='max', axis=0, copy=True, return_norm=False)
    
    print('X_test (after padding):', np.shape(X_test))
    print('y_test (after padding):', np.shape(y_test))

    input_dimension = np.shape(X_train)[1]
    print(input_dimension)
    
    reconstructed_model = keras.models.load_model("../savedModels/my_model_001")
    
    results = reconstructed_model.evaluate(X_test, y_test, batch_size=1)
    print("test loss, test acc:", results)
    
if __name__ == '__main__':
    testing()