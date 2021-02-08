# imports
import csv
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from numpy import save
from numpy import load
from sklearn.multioutput import MultiOutputClassifier
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LayerNormalization, Dense, Activation, Dropout
#from tensorflow.keras.layers import Embedding, Bidirectional, Input, LSTM, GRU, multiply, Lambda, PReLU, SimpleRNN, BatchNormalization, Conv2D, Conv1D, Flatten, LeakyReLU, MaxPooling2D, MaxPooling1D, Reshape


# read features files
y_train = load('features/train_y.npy')
features_train = load('features/features_train.npy')
features_test = load('features/features_test.npy')

X_train = features_train
y_train = y_train
X_test = features_test


# 5-fold cross validation setup
cross_validation_1_train = [x for x in range(int(X_train.shape[0]/5) ,X_train.shape[0])]
cross_validation_2_train = [x for x in range(int(X_train.shape[0]/5))]+[x for x in range(int(X_train.shape[0]*2/5) ,X_train.shape[0])]
cross_validation_3_train = [x for x in range(int(X_train.shape[0]*2/5))]+[x for x in range(int(X_train.shape[0]*3/5) ,X_train.shape[0])]
cross_validation_4_train = [x for x in range(int(X_train.shape[0]*3/5))]+[x for x in range(int(X_train.shape[0]*4/5) ,X_train.shape[0])]
cross_validation_5_train = [x for x in range(int(X_train.shape[0]*4/5))]
cross_validation_train = [cross_validation_1_train]+[cross_validation_2_train]+[cross_validation_3_train]+[cross_validation_4_train]+[cross_validation_5_train]
cross_validation_train = np.array(cross_validation_train)

cross_validation_1_test = [x for x in range(int(y_train.shape[0]/5))]
cross_validation_2_test = [x for x in range(int(y_train.shape[0]/5), int(y_train.shape[0]*2/5))]
cross_validation_3_test = [x for x in range(int(y_train.shape[0]*2/5), int(y_train.shape[0]*3/5))]
cross_validation_4_test = [x for x in range(int(y_train.shape[0]*3/5), int(y_train.shape[0]*4/5))]
cross_validation_5_test = [x for x in range(int(y_train.shape[0]*4/5), int(y_train.shape[0]))]
cross_validation_test = [cross_validation_1_test]+[cross_validation_2_test]+[cross_validation_3_test]+[cross_validation_4_test]+[cross_validation_5_test]
cross_validation_test = np.array(cross_validation_test)


# model training, with cross validation
for j in range(cross_validation_train.shape[0]):
        #fit1
    initializer = tf.keras.initializers.LecunNormal(seed=1)
    nn_model = Sequential([
        Dense(256, activation = 'selu', kernel_initializer = initializer),
        LayerNormalization(),
        Dropout(0.5),

        Dense(256, activation = 'selu', kernel_initializer = initializer),
        LayerNormalization(),
        Dropout(0.5),

        Dense(256, activation = 'selu', kernel_initializer = initializer),
        LayerNormalization(),
        Dropout(0.5),

        Dense(256, activation = 'selu', kernel_initializer = initializer),
        LayerNormalization(),
        Dropout(0.5),

        Dense(8, activation = 'sigmoid')

    ])
    #fit model

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00007, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    nn_model.compile(optimizer=optimizer, loss= tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    training_history = nn_model.fit(X_train[cross_validation_train[j]], y_train[cross_validation_train[j]],
            epochs=600,
            verbose=1)
    
    y_pred_test = nn_model.predict(X_test)
    y_pred = nn_model.predict(X_train[cross_validation_test[j]])
    y_pred_train = nn_model.predict(X_train)


    if j == 0:
        svc_results = y_pred
        svc_results_test = y_pred_test
        svc_results_train = y_pred_train

    else:
        svc_results = np.vstack((svc_results, y_pred))
        svc_results_test = (svc_results_test*j+y_pred_test)/(j+1)
        svc_results_train = (svc_results_train*j+y_pred_train)/(j+1)

    print("iteration : "+ str(j)+" Finished")


# save results
save("features/NN_features_train.npy",svc_results)
save("features/NN_features_test.npy",svc_results_test)