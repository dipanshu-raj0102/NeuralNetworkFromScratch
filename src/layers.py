import numpy as np


class DenseLayer:
  
    def __init__(self, input_dim: int, output_dim: int, name: str = "dense_layer"):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # He (Kaiming) Normal Initialization: std = sqrt(2 / input_dim)
        std = np.sqrt(2.0 / input_dim)
        self.W = np.random.randn(input_dim, output_dim).astype(np.float32) * std
        self.b = np.zeros((1, output_dim), dtype=np.float32)
        
        # Gradients (to be populated during backpropagation)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        # Cache for forward pass inputs needed during backprop
        self.X = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X  # Cache input for backward pass
        return np.matmul(X, self.W) + self.b
    
    def backward(self, dZ: np.ndarray) -> np.ndarray:

        batch_size = dZ.shape[0]
        self.dW = np.matmul(self.X.T, dZ) / batch_size
        self.db = np.sum(dZ, axis = 0, keepdims=True) / batch_size

        dX = np.matmul(dZ, self.W.T)

        return dX
    
    def __repr__(self) -> str:
        return f"DenseLayer(name='{self.name}', input_dim={self.input_dim}, output_dim={self.output_dim})"
    
