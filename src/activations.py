import numpy as np


class ReLU:
    def __init__(self, name: str = "relu"):
        
        self.name = name
        self.Z = None  # Cache inputs for backpropagation

    def forward(self, Z: np.ndarray) -> np.ndarray:
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dA: np.ndarray) -> np.ndarray:
        
        # Create a boolean mask where Z > 0, cast to float, and multiply by upstream gradient
        dZ = dA * (self.Z > 0).astype(np.float32)
        return dZ

    def __repr__(self) -> str:
        return f"ReLU(name='{self.name}')"

class Softmax:

    def __init__(self, name: str = "softmax"):
        self.name = name
        self.out = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        # Subtract max(Z) per row to prevent exp() overflow (e^x -> 1.0 max)
        shift_Z = Z - np.max(Z, axis=-1, keepdims=True)
        
        exps = np.exp(shift_Z)
        self.out = exps / np.sum(exps, axis=1, keepdims=True)
        return self.out

    def __repr__(self) -> str:
        return f"Softmax(name='{self.name}')"
