from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import numpy as np

def trained_KNN(X_train, X_test, Y_train, Y_test):
    knn = KNeighborsClassifier()
    knn_params = { 'n_neighbors': np.arange(1, 25) }
    gs = GridSearchCV(knn, knn_params, cv=5)
    gs.fit(X_train, Y_train)
    gs_best = gs.best_estimator_
    print(gs.best_params_)

    yhat = gs_best.predict(X_test)

    print(classification_report(Y_test, yhat))
    metrics.ConfusionMatrixDisplay.from_estimator(gs_best, X_test, Y_test)

    return gs_best

def trained_LogisticRegression(X_train, X_test, Y_train, Y_test):
    lr = LogisticRegression()
    lr_params = {'C': np.arange(0.1, 1, 0.1), 'solver' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}
    gs = GridSearchCV(lr, lr_params, cv=5)
    gs.fit(X_train, Y_train)
    gs_best = gs.best_estimator_
    print(gs.best_params_)

    yhat = gs_best.predict(X_test)

    print(classification_report(Y_test, yhat))
    metrics.ConfusionMatrixDisplay.from_estimator(gs_best, X_test, Y_test)

    return gs_best

def trained_RandomForestClassifier(X_train, X_test, Y_train, Y_test):
    rfc = RandomForestClassifier()
    rfc_params = {'n_estimators': np.arange(100, 200, 10), 'max_depth': np.arange(1,20)}
    gs = GridSearchCV(rfc, rfc_params, cv=5)
    gs.fit(X_train, Y_train)
    gs_best = gs.best_estimator_
    print(gs.best_params_)

    yhat = gs_best.predict(X_test)

    print(classification_report(Y_test, yhat))
    metrics.ConfusionMatrixDisplay.from_estimator(gs_best, X_test, Y_test)

    return gs_best
