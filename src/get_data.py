import os

from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from ruamel.yaml import YAML


yaml = YAML(typ="safe")

def get_data():

    params = yaml.load(
        open("params.yaml", encoding="utf-8")
    )

    digits = datasets.load_digits()

    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        digits.target,
        test_size=params["get-raw-data"]["test_size"],
        shuffle=params["get-raw-data"]["shuffle"]
    )

    try:
        os.mkdir("/home/thomas/projects/tutorials/dvc_own_test/data/")
    except:
        pass

    try:
        os.mkdir("/home/thomas/projects/tutorials/dvc_own_test/data/train/")
    except:
        pass

    try:
        os.mkdir("/home/thomas/projects/tutorials/dvc_own_test/data/test/")
    except:
        pass

    np.savetxt(
        fname="/home/thomas/projects/tutorials/dvc_own_test/data/train/X.csv",
        X=X_train,
        delimiter=";"
    )

    np.savetxt(
        fname="/home/thomas/projects/tutorials/dvc_own_test/data/train/y.csv",
        X=y_train,
        delimiter=";"
    )

    np.savetxt(
        fname="/home/thomas/projects/tutorials/dvc_own_test/data/test/X.csv",
        X=X_test,
        delimiter=";"
    )

    np.savetxt(
        fname="/home/thomas/projects/tutorials/dvc_own_test/data/test/y.csv",
        X=y_test,
        delimiter=";"
    )


if __name__ == "__main__":
    get_data()
