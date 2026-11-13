# STEP 2: Look at the dataset and reproduce the GitHub project's starting point.
# Run it with:  python step2_explore_data.py

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import load_data

# ---- Part A: what is inside the dataset? ----
dataset = load_breast_cancer()
print("Number of patients (rows):  ", dataset.data.shape[0])
print("Number of measurements:     ", dataset.data.shape[1])
print("First 5 measurement names:  ", [str(n) for n in dataset.feature_names[:5]])
print("Class names:                ", [str(n) for n in dataset.target_names])
print("Malignant count:            ", np.sum(dataset.target == 0))
print("Benign count:               ", np.sum(dataset.target == 1))

# ---- Part B: our train/test split ----
X_train, X_test, y_train, y_test = load_data()
print("\nTraining set:", X_train.shape, " Test set:", X_test.shape)
print("Labels used:", np.unique(y_train))

# ---- Part C: the GitHub project's first step (SVM with default settings) ----
model = SVC()                       # default C=1.0, gamma='scale'
model.fit(X_train, y_train)
acc_raw = model.score(X_test, y_test)
print(f"\nDefault SVM, unscaled data:  test accuracy = {acc_raw:.4f}")

# Same model, but with the measurements scaled to similar ranges first.
scaler = StandardScaler().fit(X_train)          # learn the scaling from TRAINING data only
model.fit(scaler.transform(X_train), y_train)
acc_scaled = model.score(scaler.transform(X_test), y_test)
print(f"Default SVM, scaled data:    test accuracy = {acc_scaled:.4f}")
print("\nScaling matters! From now on we always scale the data.")