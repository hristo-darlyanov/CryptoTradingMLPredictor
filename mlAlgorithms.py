from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import metrics
import numpy as np

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
    lr = LogisticRegression(class_weight={0:1,1:5})
    lr_params = {'C': np.arange(0.1, 1, 0.1), 'solver' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}
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
    rfc_params = {'n_estimators': np.arange(100, 200, 10), 'max_depth': np.arange(1,2)}
    gs = GridSearchCV(rfc, rfc_params, cv=5)
    gs.fit(X_train, Y_train)
    gs_best = gs.best_estimator_
    print(gs.best_params_)

    yhat = gs_best.predict(X_test)
    yhat_train = gs_best.predict(X_train)

    print(classification_report(Y_test, yhat))
    print("RFC TEST SET -", accuracy_score(Y_test, yhat))
    print("RFC TRAIN SET -", accuracy_score(Y_train, yhat_train))
    metrics.ConfusionMatrixDisplay.from_estimator(gs_best, X_test, Y_test)

    return gs_best

def trained_VotingClassifier(lr_model, knn_model, X_train, X_test, Y_train, Y_test):
    estimators = [('lr', lr_model), ('knn', knn_model)]
    
    vc = VotingClassifier(estimators=estimators, voting='hard')
    vc.fit(X_train, Y_train)

    yhat = vc.predict(X_test)

    print("VC TEST SET -", accuracy_score(Y_test, yhat))
    metrics.ConfusionMatrixDisplay.from_estimator(vc, X_test, Y_test)

    return vc
    