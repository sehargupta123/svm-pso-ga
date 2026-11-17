# STEP 3: Test our own SVM and compare it with scikit-learn's SVM.
# Run it with:  python step3_test_my_svm.py
# Part A: our QP solver (cvxopt) vs scikit-learn. If the accuracies match, our SVM is correct.
# Part B: our SMO solver vs our QP solver vs scikit-learn.
# We compare on a validation part of the TRAINING data; the real test set is not used here.

import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import load_data
from my_svm import MySVM

X_train, X_test, y_train, y_test = load_data()
X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=0)

# Scale the data (learn the scaling from the fitting part only).
scaler = StandardScaler().fit(X_fit)
X_fit_s = scaler.transform(X_fit)
X_val_s = scaler.transform(X_val)

settings = [(1, 0.01), (1, 0.1), (10, 0.01), (100, 0.001), (0.1, 0.1)]

print("PART A: QP solver vs scikit-learn")
print(f"{'C':>6} {'gamma':>7} | {'MySVM acc':>9} {'sklearn acc':>11} | {'MySVM SVs':>9} {'sklearn SVs':>11} | {'KKT ok %':>8} {'time (s)':>8}")
print("-" * 88)
for C, gamma in settings:
    start = time.time()
    mine = MySVM(C=C, gamma=gamma).fit(X_fit_s, y_fit)
    seconds = time.time() - start

    ref = SVC(C=C, gamma=gamma, kernel="rbf").fit(X_fit_s, y_fit)

    print(f"{C:>6} {gamma:>7} | {mine.score(X_val_s, y_val):>9.4f} {ref.score(X_val_s, y_val):>11.4f} | "
          f"{mine.n_support:>9} {ref.n_support_.sum():>11} | {mine.check_kkt():>8.1f} {seconds:>8.2f}")

print("\nPART B: QP vs SMO vs scikit-learn (validation accuracy)")
print(f"{'C':>6} {'gamma':>7} | {'QP acc':>7} {'SMO acc':>8} {'sklearn':>8} | {'QP SVs':>6} {'SMO SVs':>7} | {'SMO KKT %':>9} {'SMO time':>8}")
print("-" * 86)
n_close = 0
for C, gamma in settings:
    qp = MySVM(C=C, gamma=gamma, solver="qp").fit(X_fit_s, y_fit)
    start = time.time()
    smo = MySVM(C=C, gamma=gamma, solver="smo").fit(X_fit_s, y_fit)
    seconds = time.time() - start
    ref = SVC(C=C, gamma=gamma, kernel="rbf").fit(X_fit_s, y_fit)
    a_qp, a_smo, a_ref = qp.score(X_val_s, y_val), smo.score(X_val_s, y_val), ref.score(X_val_s, y_val)
    n_close += abs(a_smo - a_ref) <= 0.02       # within 2 percentage points (3 patients of 114)
    print(f"{C:>6} {gamma:>7} | {a_qp:>7.4f} {a_smo:>8.4f} {a_ref:>8.4f} | {qp.n_support:>6} {smo.n_support:>7} | "
          f"{smo.check_kkt():>9.1f} {seconds:>8.2f}")
print(f"\nSMO is within 2 percentage points of scikit-learn in {int(n_close)} of {len(settings)} settings.")
print("(Simplified SMO stops early, so it is a little less exact than the QP solver. That is expected.)")
