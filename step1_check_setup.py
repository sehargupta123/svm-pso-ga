# STEP 1: Check that Python and all the libraries are installed correctly.
# Run it with:  python step1_check_setup.py

import sys
print("Python version:", sys.version.split()[0])

import numpy
import scipy
import sklearn
import cvxopt
import matplotlib
import pandas

print("numpy       ", numpy.__version__)
print("scipy       ", scipy.__version__)
print("scikit-learn", sklearn.__version__)
print("cvxopt      ", cvxopt.__version__)
print("matplotlib  ", matplotlib.__version__)
print("pandas      ", pandas.__version__)

# These two are only needed for the optional steps (Bayesian baseline in step 6, NSGA-II in step 10).
for name in ("optuna", "pymoo"):
    try:
        module = __import__(name)
        print(f"{name:<12}", module.__version__, "(optional)")
    except ImportError:
        print(f"{name:<12}", "NOT installed (optional)  ->  pip install", name)

print("\nAll required libraries are installed. You are ready for Step 2!")
