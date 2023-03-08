from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import numpy as np

def trained_KNN(X_train, X_test, Y_train, Y_test):
    knn = KNeighborsClassifier()
    knn_params = { 'n_neighbors': np.arange(1, 25) }
    gs = GridSearchCV(knn, knn_params)
    gs.fit(X_train, Y_train)
    gs_best = gs.best_estimator_

    yhat = gs_best.predict(X_test)

    print(classification_report(Y_test, yhat))
    metrics.ConfusionMatrixDisplay.from_estimator(gs_best, X_test, Y_test)

    return gs_best
