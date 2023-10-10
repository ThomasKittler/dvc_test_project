from sklearn import svm
from ruamel.yaml import YAML
import numpy as np
from joblib import dump, load


yaml = YAML(typ="safe")


def train_clf():
    params = yaml.load(
        open("params.yaml", encoding="utf-8")
    )

    X_train = np.loadtxt(
        fname="data/train/X.csv",
        delimiter=";"
    )

    y_train = np.loadtxt(
        fname="data/train/y.csv",
        delimiter=";"
    )

    clf = svm.SVC(gamma=params["train"]["gamma"])

    clf.fit(X_train, y_train)

    dump(clf, "models/trained_model.joblib")

if __name__ == "__main__":
    train_clf()
