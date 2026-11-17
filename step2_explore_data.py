# STEP 2: Look at the dataset and reproduce the GitHub project's starting point.
# Run it with:  python step2_explore_data.py
# (The real test set stays untouched: the model comparison below uses a validation part
#  carved out of the TRAINING data.)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import load_data

# ---- Part A: what is inside the dataset? ----
X_train, X_test, y_train, y_test = load_data()
print("Number of training + test samples:", len(y_train) + len(y_test))
print("Number of measurements:           ", X_train.shape[1])
print("Malignant (label -1) count:       ", int(np.sum(y_train == -1) + np.sum(y_test == -1)))
print("Benign    (label +1) count:       ", int(np.sum(y_train == 1) + np.sum(y_test == 1)))

# ---- Part B: our train/test split ----
print("\nTraining set:", X_train.shape, " Test set:", X_test.shape, "(test set is locked away until the end)")
print("Labels used:", np.unique(y_train))

# ---- Part C: the GitHub project's first step (SVM with default settings) ----
# We split the training data again: 75% to fit, 25% to check.
X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=0)

model = SVC()                       # default C=1.0, gamma='scale'
model.fit(X_fit, y_fit)
acc_raw = model.score(X_val, y_val)
print(f"\nDefault SVM, unscaled data:  validation accuracy = {acc_raw:.4f}")

# Same model, but with the measurements scaled to similar ranges first.
scaler = StandardScaler().fit(X_fit)            # learn the scaling from the fitting part only
model.fit(scaler.transform(X_fit), y_fit)
acc_scaled = model.score(scaler.transform(X_val), y_val)
print(f"Default SVM, scaled data:    validation accuracy = {acc_scaled:.4f}")
print("\nScaling matters! From now on we always scale the data.")
