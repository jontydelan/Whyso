import pandas as pd
import numpy as np
import math, os, sys
import time

import seaborn as sns
import matplotlib.pyplot as plt

from functools import partial

sys.path.insert(0, r"C:\Users\M1049231\Dev\Equifax")
from utils import efx_utils

sns.set_style(style="darkgrid")
# sns.set_context("notebook")
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    f1_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_oversampled_ds(n=2):
    actifact = efx_utils._load_all()
    FEAT = actifact["FINAL_FEAT_V2"]

    df = pd.DataFrame(actifact["X"][FEAT], columns=FEAT)
    df["y"] = actifact["y"]
    print("original dataset : ", df.shape)
    df2 = df[df.y == 1]
    for i in range(n):
        df = pd.concat([df, df2], ignore_index=True)
    print("oversampled dataset :  ", df.shape)
    X = df[FEAT].values
    y = df["y"].values
    return (X, y)


def get_dataset(oversample=False, *args, **kwargs):
    actifact = efx_utils._load_all()
    FEAT = actifact["FINAL_FEAT_V2"]
    # FEAT.remove("HAS_DEFAULT")

    if oversample:
        (X, y) = _get_oversampled_ds(*args, **kwargs)
    else:
        X = actifact["X"][FEAT].values
        y = actifact["y"]

    return _make_dataset(X, y)


def _make_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)

    y_test = y_test.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    print("ytrain shape : ", y_train.shape)

    return (X_train, X_test, y_train, y_test)


def get_config(input, res: dict = dict(), epochs=10, lr=0.0001, activation="sig"):
    """configurations of the nn model"""

    if "config" not in res.keys():
        config = dict(
            input=input, output=1, lr=lr, epochs=epochs, activation=activation
        )
    else:
        config = res["config"]

    return config


class Linear_Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg["input"], cfg["output"], bias=False)
        print(f"using {self.cfg['activation']} activation ")

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        #         x = self.fc1(x)
        x = torch.sigmoid(x)
        #         x = nn.BCEWithLogitsLoss(x)
        return x

    def predict(self, X_test, y_test):

        with torch.no_grad():
            self.eval()
            y_pred = self(X_test).detach().numpy().round()
            assert (
                y_pred.shape == y_test.shape
            ), f"shape not good y_pred.shape {y_pred.shape} y_test.shape {y_test.shape} "
            acc = round(accuracy_score(y_test, y_pred), 3)
            self.train()

        return acc


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg["input"], 2 * cfg["input"], bias=True)
        self.fc2 = nn.Linear(2 * cfg["input"], cfg["output"], bias=True)
        print(f"using {self.cfg['activation']} activation : conf using relu +sig + bia")

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #         x = self.fc1(x)
        #         x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def predict(self, X_test, y_test):

        with torch.no_grad():
            self.eval()
            y_pred = self(X_test).detach().numpy().round()
            assert (
                y_pred.shape == y_test.shape
            ), f"shape not good y_pred.shape {y_pred.shape} y_test.shape {y_test.shape} "
            acc = round(accuracy_score(y_test, y_pred), 3)
            self.train()

        return acc
