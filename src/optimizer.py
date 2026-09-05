

class SGDOptimizer:
    def __init__(self, layers: list, lr: float = 0.01):
      
        self.layers = layers
        self.lr = lr

    def step(self) -> None:
      
        for layer in self.layers:
            if hasattr(layer, "W") and hasattr(layer, "b"):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db


