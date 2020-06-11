import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _sigmoid_calibration


def Z_score(p, y):
    # Calibration check following Turner et al. (2019).
    # http://arxiv.org/abs/1811.11357
    # If the predictions are well calibrated, then:
    #   y[i] ~ Bern(p[i])
    # And so Z, as calculated below, should be ~Norm(0,1).
    residuals = y - p
    variance = p * (1 - p)
    Z = np.sum(residuals) / np.sqrt(np.sum(variance))
    return Z


class Beta(object):
    """
    Beta calibration (via logisitic regression).
    Kull et al. (2017), http://proceedings.mlr.press/v54/kull17a.html
    """

    def fit(self, x, y):

        # Exclude predictions that are too confident.
        # They cause trouble with the logistic regression.
        idx = np.where(np.bitwise_and(0 < x, x < 1))[0]
        if len(x) != len(idx):
            warnings.warn(
                "BetaCalibration: excluding values at the edge of support "
                f"({len(x)-len(idx)} of {len(x)})."
            )
        x = x[idx]
        y = y[idx]

        s_a = np.log(x)
        s_b = -np.log(1.0 - x)
        s = np.column_stack([s_a, s_b])
        lr = LogisticRegression().fit(s, y)
        a, b, c = lr.coef_[0][0], lr.coef_[0][1], lr.intercept_[0]
        # Should give a>=0 and b>=0. Kull et al. suggest that if this is found
        # to be untrue, the negative parameter should be fixed at zero and
        # the other parameters refit.
        if a < 0:
            warnings.warn(f"BetaCalibration: a<0 (a={a:.3g}), refitting.")
            a = 0
            lr = LogisticRegression().fit(s_b.reshape(-1, 1), y)
            b, c = lr.coef_[0][0], lr.intercept_[0]
        elif b < 0:
            warnings.warn(f"BetaCalibration: b<0 (b={b:.3g}), refitting.")
            b = 0
            lr = LogisticRegression().fit(s_a.reshape(-1, 1), y)
            a, c = lr.coef_[0][0], lr.intercept_[0]
        self.a = a
        self.b = b
        self.c = c
        return self

    def predict(self, x):
        u = np.exp(self.c) * x ** self.a
        v = (1 - x) ** self.b
        return u / (u + v)


class Isotonic(object):
    """
    Isotonic regression.
    """

    def __init__(self):
        self.ir = IsotonicRegression(out_of_bounds="clip")

    def fit(self, x, y):
        self.ir.fit(x, y)
        return self

    def predict(self, x):
        return self.ir.predict(x)


class Platt(object):
    """
    Platt scaling.
    https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/calibration.py#L403
    """

    def fit(self, x, y):
        self.a, self.b = _sigmoid_calibration(x, y)
        return self

    def predict(self, x):
        z = np.exp(-self.a * x - self.b)
        return z / (z + 1)


calibration_classes = (Beta, Isotonic, Platt)
