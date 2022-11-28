import sklearn

def f1_score(y_test, y_pred):
    return sklearn.metrics.f1_score(y_test, y_pred, average="micro")