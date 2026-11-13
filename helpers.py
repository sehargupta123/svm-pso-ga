# helpers.py
# Small shared helper: train on ALL training data with the chosen (C, gamma) and score on the TEST set.

from sklearn.preprocessing import StandardScaler
from my_svm import MySVM


def test_accuracy(X_train, y_train, X_test, y_test, log_C, log_gamma):
    scaler = StandardScaler().fit(X_train)
    model = MySVM(C=10 ** log_C, gamma=10 ** log_gamma)
    model.fit(scaler.transform(X_train), y_train)
    return model.score(scaler.transform(X_test), y_test)