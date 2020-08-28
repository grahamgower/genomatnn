import collections
import logging
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _sigmoid_calibration

logger = logging.getLogger(__name__)


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
        lr = LogisticRegression(solver="lbfgs").fit(s, y)
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
        x = x.astype(np.float32)
        y = y.astype(np.float32)
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


def groupby_indexes(a):
    indexes = collections.defaultdict(list)
    for j, k in enumerate(a):
        indexes[k].append(j)
    return list(indexes.keys()), [np.array(v) for v in indexes.values()]


def resample_indexes(a, weights=None, rng=None):
    """
    Return indexes that would resample array ``a`` to have unique category
    proportions matching the given weights.
    All samples are retained from the smallest category, and other
    categories may be up- or down-sampled, depending on the the desired
    weights.
    """
    if rng is None:
        rng = np.random.default_rng(1234)
    unique, indexes = groupby_indexes(a)
    if weights is None:
        weights = {m: 1 for m in unique}
    w = np.array([weights[m] for m in unique])
    q = np.array([len(ind) for ind in indexes])

    # Given vector of counts, q, we want to achieve proportions w,
    # while retaining all samples in our smallest category.
    # j is the index of our constraining category.
    j = (q * w).argmin()
    # We'll resample to p[i] for each category i.
    p = (q[j] * w / w[j]).round().astype(int)
    assert p[j] == q[j]
    logger.debug(f"Resampling with counts {p}.")

    upidx = []
    for i in range(len(unique)):
        # Sample with replacement only if we need more than are available
        replace = p[i] > q[i]
        ind = rng.choice(indexes[i], size=p[i], replace=replace)
        upidx.append(ind)

    return np.hstack(upidx)


def calibrate(conf, labels, metadata, pred, cal=None):
    weights = conf.get("calibrate.weights")
    upidx = resample_indexes(metadata["modelspec"], weights)
    logger.info(f"Fitting {conf.calibration.__name__} calibration")
    if cal is None:
        assert conf.calibration is not None
        cal = conf.calibration()
    return cal.fit(pred[upidx], labels[upidx])
