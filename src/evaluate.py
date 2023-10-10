from sklearn import metrics
from joblib import load
from ruamel.yaml import YAML
import numpy as np


yaml = YAML(typ="safe")

def evaluate_clf():
    params = yaml.load(
        open("params.yaml", encoding="utf-8")
    )

    clf = load("models/trained_model.joblib")

    X_test = np.loadtxt(
        fname="data/test/X.csv",
        delimiter=";"
    )

    y_test = np.loadtxt(
        fname="data/test/y.csv",
        delimiter=";"
    )

    predicted = clf.predict(X_test)

    clf_report = metrics.classification_report(y_test, predicted)
