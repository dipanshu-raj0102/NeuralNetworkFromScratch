import numpy as np

from activations import *
from layers import *
from losses import *


class MLP:

    def __init__(self, input_dim: int = 784, num_classes: int = 26):
        self.layer1 = DenseLayer(input_dim, 128, name="layer1")
        self.relu1 = ReLU(name="relu1")

        self.layer2 = DenseLayer(128, 64, name="layer2")
        self.relu2 = ReLU(name="relu2")

        self.layer3 = DenseLayer(64, num_classes, name="Output layer")
        self.softmax = Softmax(name="softmax")

        self.loss_function = CrossEntropyLoss()
        self.y_pred = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        z1 = self.layer1.forward(X)
        a1 = self.relu1.forward(z1)

        z2 = self.layer2.forward(a1)
        a2 = self.relu2.forward(z2)

        z3 = self.layer3.forward(a2)
        self.y_pred = self.softmax.forward(z3)

        return self.y_pred

    def compute_loss(self, y_true: np.ndarray) -> float:
        return self.loss_function.forward(y_true, self.y_pred)

    def backward(self, y_true: np.ndarray) -> None:
        dZ3 = self.loss_function.backward(y_true, self.y_pred)

        dA2 = self.layer3.backward(dZ3)

        dZ2 = self.relu2.backward(dA2)
        dA1 = self.layer2.backward(dZ2)

        dZ1 = self.relu1.backward(dA1)
        self.layer1.backward(dZ1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
