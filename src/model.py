import numpy as np

from activations import *
from layers import *
from losses import *


class MLP:

    def __init__(self, input_dim: int = 784, num_classes: int = 26):
        # Layer 1: 784 -> 256
        self.layer1 = DenseLayer(input_dim, 256, name="layer1")
        self.relu1 = ReLU(name="relu1")

        # Layer 2: 256 -> 128
        self.layer2 = DenseLayer(256, 128, name="layer2")
        self.relu2 = ReLU(name="relu2")

        # Layer 3: 128 -> 64
        self.layer3 = DenseLayer(128, 64, name="layer3")
        self.relu3 = ReLU(name="relu3")

        # Layer 4 (Output): 64 -> 26
        self.layer4 = DenseLayer(64, num_classes, name="Output layer")
        self.softmax = Softmax(name="softmax")

        self.loss_function = CrossEntropyLoss()
        self.y_pred = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        z1 = self.layer1.forward(X)
        a1 = self.relu1.forward(z1)

        z2 = self.layer2.forward(a1)
        a2 = self.relu2.forward(z2)

        z3 = self.layer3.forward(a2)
        a3 = self.relu3.forward(z3)

        z4 = self.layer4.forward(a3)
        self.y_pred = self.softmax.forward(z4)

        return self.y_pred

    def compute_loss(self, y_true: np.ndarray) -> float:
        return self.loss_function.forward(y_true, self.y_pred)

    def backward(self, y_true: np.ndarray) -> None:
        # Step 1: Gradient from Softmax + CrossEntropy w.r.t Z4
        dZ4 = self.loss_function.backward(y_true, self.y_pred)

        # Step 2: Layer 4 (64 -> 26)
        dA3 = self.layer4.backward(dZ4)

        # Step 3: Layer 3 (128 -> 64)
        dZ3 = self.relu3.backward(dA3)
        dA2 = self.layer3.backward(dZ3)

        # Step 4: Layer 2 (256 -> 128)
        dZ2 = self.relu2.backward(dA2)
        dA1 = self.layer2.backward(dZ2)

        # Step 5: Layer 1 (784 -> 256)
        dZ1 = self.relu1.backward(dA1)
        self.layer1.backward(dZ1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
