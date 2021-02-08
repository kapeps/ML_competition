# imports
from sklearn.metrics import log_loss
import csv

from sklearn import preprocessing
import pandas as pd
import numpy as np
from numpy import save
from numpy import load
from sklearn.multioutput import MultiOutputClassifier
from catboost import CatBoostClassifier


# read features files
y_train = load('features/train_y.npy')
features_train = load('features/features_train.npy')
features_test = load('features/features_test.npy')
features_NNCV_train = load('features/NN_features_train.npy')
features_NNCV_test = load('features/NN_features_test.npy')
features_SVC_train = load('features/SVC_features_train.npy')
features_SVC_test = load('features/SVC_features_test.npy')
features_C_train = load('features/Cat_features_train.npy')
features_C_test = load('features/Cat_features_test.npy')
features_RFE_train = load('features/TREE_features_train.npy')
features_RFE_test = load('features/TREE_features_test.npy')
features_LOG_train = load('features/LOG_features_train.npy')
features_LOG_test = load('features/LOG_features_test.npy')
features_RF_train = load('features/RF_features_train.npy')
features_RF_test = load('features/RF_features_test.npy')

features_train = np.concatenate((features_train, features_NNCV_train), axis=1)
features_train = np.concatenate((features_train, features_SVC_train), axis=1)
features_train = np.concatenate((features_train, features_C_train), axis=1)
features_train = np.concatenate((features_train, features_RFE_train), axis=1)
features_train = np.concatenate((features_train, features_LOG_train), axis=1)
features_train = np.concatenate((features_train, features_RF_train), axis=1)


features_test = np.concatenate((features_test, features_NNCV_test), axis=1)
features_test = np.concatenate((features_test, features_SVC_test), axis=1)
features_test = np.concatenate((features_test, features_C_test), axis=1)
features_test = np.concatenate((features_test, features_RFE_test), axis=1)
features_test = np.concatenate((features_test, features_LOG_test), axis=1)
features_test = np.concatenate((features_test, features_RF_test), axis=1)

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
    model = CatBoostClassifier(iterations=2000, od_type = "IncToDec",task_type="GPU", loss_function="Logloss", verbose=1)
    clf = MultiOutputClassifier(model).fit(X_train[cross_validation_train[j]], y_train[cross_validation_train[j]])

    y_pred = (np.array(clf.predict_proba(X_train[cross_validation_test[j]])).T)[1]
    y_pred_test = (np.array(clf.predict_proba(X_test)).T)[1]
    y_pred_train = (np.array(clf.predict_proba(X_train)).T)[1]

    print(y_pred_train.shape)
    if j == 0:
        svc_results = y_pred
        svc_results_test = y_pred_test
        svc_results_train = y_pred_train
    else:
        svc_results = np.vstack((svc_results, y_pred))
        svc_results_test = (svc_results_test*j+y_pred_test)/(j+1)
        svc_results_train = (svc_results_train*j+y_pred_train)/(j+1)


    print("iteration : "+ str(j)+" Finished")


# log loss calculation
loss_sum = 0
for i in range(8):
    loss_sum += log_loss(y_train[:,i],svc_results[:,i])

print("CV Training loss : "+ str(loss_sum/8))

loss_sum = 0
for i in range(8):
    loss_sum += log_loss(y_train[:,i],svc_results_train[:,i])

print("Training loss : "+ str(loss_sum/8))


# generating submission file
pred_df = pd.DataFrame(svc_results_test, columns=['updates', 'personal', 'promotions',
                'forums', 'purchases', 'travel',
                'spam', 'social'])
pred_df.to_csv("y_FINAL.csv", index=True, index_label='Id')  

