import numpy as np


class CrossEntropyLoss:

    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred_clipped = np.clip(y_pred, self.eps, 1.0)
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
        return float(loss)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        m = y_true.shape[0]
        dZ = (y_pred - y_true) / m
        return dZ



