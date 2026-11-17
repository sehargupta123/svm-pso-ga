# helpers.py
# Small shared helpers.
# test_report / test_accuracy are the ONLY places in the project that touch the test set.
# They train on ALL training data with the chosen (C, gamma) and score on the TEST set.

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from my_svm import MySVM

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")     # always next to the code, whatever folder you run from
FIGURES_DIR = os.path.join(HERE, "figures")


def make_output_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def test_report(X_train, y_train, X_test, y_test, log_C, log_gamma, use_my_svm=True):
    """Final model on the test set. Returns accuracy, confusion matrix counts and malignant recall.

    Malignant = label -1. tp = malignant caught, fn = malignant MISSED,
    fp = benign wrongly called malignant, tn = benign correct.
    """
    scaler = StandardScaler().fit(X_train)          # scaler learns from training data only
    C, gamma = 10 ** log_C, 10 ** log_gamma
    model = MySVM(C=C, gamma=gamma) if use_my_svm else SVC(C=C, gamma=gamma, kernel="rbf")
    model.fit(scaler.transform(X_train), y_train)
    pred = model.predict(scaler.transform(X_test))
    tp = int(np.sum((y_test == -1) & (pred == -1)))
    fn = int(np.sum((y_test == -1) & (pred == 1)))
    fp = int(np.sum((y_test == 1) & (pred == -1)))
    tn = int(np.sum((y_test == 1) & (pred == 1)))
    return {"acc": float(np.mean(pred == y_test)), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": tp / (tp + fn) if tp + fn else float("nan")}


def test_accuracy(X_train, y_train, X_test, y_test, log_C, log_gamma, use_my_svm=True):
    return test_report(X_train, y_train, X_test, y_test, log_C, log_gamma, use_my_svm)["acc"]


def test_accuracy_model(X_train, y_train, X_test, y_test, model):
    """Same idea as test_accuracy, but for any scikit-learn model (used by step 9: other kernels)."""
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(StandardScaler(), model).fit(X_train, y_train)    # scaler learns from training data only
    return float(np.mean(pipe.predict(X_test) == y_test))
