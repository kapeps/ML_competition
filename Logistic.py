# imports
from sklearn.metrics import log_loss
import csv

import pandas as pd
import numpy as np
from numpy import save
from numpy import load
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression


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
    model = LogisticRegression(random_state=0, verbose = False, max_iter = 10000,tol=1e-4, C = 100, solver = "newton-cg")
    clf = MultiOutputClassifier(model).fit(X_train[cross_validation_train[j]], y_train[cross_validation_train[j]])

    y_pred = (np.array(clf.predict_proba(X_train[cross_validation_test[j]])).T)[1]
    y_pred_test = (np.array(clf.predict_proba(X_test)).T)[1]
    y_pred_train = (np.array(clf.predict_proba(X_train)).T)[1]


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


# save results
save("features/LOG_features_train.npy",svc_results)
save("features/LOG_features_test.npy",svc_results_test)