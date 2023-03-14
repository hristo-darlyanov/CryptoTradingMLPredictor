from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, r2_score, mean_squared_error
from sklearn import metrics
from sklearn.feature_selection import RFECV
import numpy as np
import matplotlib.pyplot as plt

def trained_LinearRegression(X_train, X_test, Y_train, Y_test):
    lr = LinearRegression()
    lr.fit(X_train, Y_train)

    yhat = lr.predict(X_test)

    print("LR TEST SET -", r2_score(Y_test, yhat))
    print("LR MSE -", mean_squared_error(Y_test, yhat))

    return lr


def trained_KNN(X_train, X_test, Y_train, Y_test):
    knn = KNeighborsClassifier()
    knn_params = { 'n_neighbors': np.arange(1, 200) }
    gs = GridSearchCV(knn, knn_params, cv=5)
    gs.fit(X_train, Y_train)
    gs_best = gs.best_estimator_
    print(gs.best_params_)

    yhat = gs_best.predict(X_test)
    yhat_train = gs_best.predict(X_train)
    
    print(classification_report(Y_test, yhat))
    print("KNN TEST SET -", accuracy_score(Y_test, yhat))
    print("KNN TRAIN SET -", accuracy_score(Y_train, yhat_train))
    metrics.ConfusionMatrixDisplay.from_estimator(gs_best, X_test, Y_test)

    return gs_best

def trained_LogisticRegression(X_train, X_test, Y_train, Y_test):
    lr = LogisticRegression(class_weight={0:1, 1:1.2})
    lr_params = {'C' : np.arange(0.1, 1, 0.1), 'solver' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}
    gs = GridSearchCV(lr, lr_params, cv=5)
    gs.fit(X_train, Y_train)
    gs_best = gs.best_estimator_
    print(gs.best_params_)

    yhat = gs_best.predict(X_test)
    yhat_train = gs_best.predict(X_train)
    
    print(classification_report(Y_test, yhat))
    print("LR TEST SET -", accuracy_score(Y_test, yhat))
    print("LR TRAIN SET -", accuracy_score(Y_train, yhat_train))
    metrics.ConfusionMatrixDisplay.from_estimator(gs_best, X_test, Y_test)

    return gs_best

def trained_RandomForestClassifier(X_train, X_test, Y_train, Y_test):
    #best max_depth determined to be 1
    #will not be adding it to the GridSearchCV because its taking too long to compute
    rfc = RandomForestClassifier()
    #rfc_params = {'n_estimators': np.arange(100, 200, 10), 'max_depth': np.arange(1,2)}
    #gs = GridSearchCV(rfc, rfc_params, cv=5)
    #gs.fit(X_train, Y_train)
    #gs_best = gs.best_estimator_
    #print(gs.best_params_)

    rfecv = RFECV(estimator=rfc, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
    rfecv = rfecv.fit(X_train, Y_train)

    yhat = rfecv.predict(X_test)
    yhat_train = rfecv.predict(X_train)

    print(classification_report(Y_test, yhat))
    print("RFC TEST SET -", accuracy_score(Y_test, yhat))
    print("RFC TRAIN SET -", accuracy_score(Y_train, yhat_train))
    metrics.ConfusionMatrixDisplay.from_estimator(rfecv, X_test, Y_test)

    print('Optimal number of features :', rfecv.n_features_)

    return rfecv

def trained_VotingClassifier(lr_model, knn_model, X_train, X_test, Y_train, Y_test):
    estimators = [('lr', lr_model), ('knn', knn_model)]
    
    vc = VotingClassifier(estimators=estimators, voting='hard')
    vc.fit(X_train, Y_train)

    yhat = vc.predict(X_test)

    print("VC TEST SET -", accuracy_score(Y_test, yhat))
    metrics.ConfusionMatrixDisplay.from_estimator(vc, X_test, Y_test)

    return vc