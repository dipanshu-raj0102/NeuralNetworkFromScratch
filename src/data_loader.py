import gzip
import struct

import numpy as np

# IDX magic numbers
IMAGE_MAGIC = 2051
LABEL_MAGIC = 2049


def load_images(path: str) -> np.ndarray:
    """Load EMNIST image file (.idx3-ubyte.gz)

    Returns:
        images: ndarray of shape (N, 28, 28), dtype=uint8
    """
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != IMAGE_MAGIC:
            raise ValueError(f"Invalid image file. Magic number: {magic}")

        num_images = struct.unpack(">I", f.read(4))[0]
        rows = struct.unpack(">I", f.read(4))[0]
        cols = struct.unpack(">I", f.read(4))[0]

        buffer = f.read(rows * cols * num_images)

        images = np.frombuffer(buffer, dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)

    return images


def load_labels(path: str) -> np.ndarray:
    """Load EMNIST label file (.idx1-ubyte.gz)

    Returns:
        labels: ndarray of shape (N,), values 0-25
    """
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != LABEL_MAGIC:
            raise ValueError(f"Invalid label file. Magic number: {magic}")

        num_labels = struct.unpack(">I", f.read(4))[0]

        buffer = f.read(num_labels)

        labels = np.frombuffer(buffer, dtype=np.uint8)

    return labels - 1


def normalize(X):
    """
    Flattens 2D images to 1D vectors and scales pixel values from [0, 255] to [0.0, 1.0].
    
    Input:  (N, 28, 28) uint8
    Output: (N, 784) float32
    """
    X_flat = X.reshape(X.shape[0], -1).astype(np.float32)
    return X_flat / 255.0


def one_hot(y, num_classes=26):
    """
    Converts integer class labels into 1-hot encoded binary vectors.
    
    Input:  (N,) integers in range [0, 25]
    Output: (N, 26) float32 matrix
    """
    N = y.shape[0]
    y_one_hot = np.zeros((N, num_classes), dtype=np.float32)
    y_one_hot[np.arange(N), y] = 1.0
    return y_one_hot

def train_cv_split(X, y, cv_ratio=0.1, seed=42):
    """
    Shuffles data and splits it into training and validation sets.
    
    Input:  X: (N, 784), y: (N, 26) or (N,)
    Output: X_train, y_train, X_val, y_val
    """
    if seed is not None:
        np.random.seed(seed)
        
    N = X.shape[0]
    indices = np.random.permutation(N)
    
    cv_size = int(N * cv_ratio)
    cv_indices = indices[:cv_size]
    train_indices = indices[cv_size:]
    
    return X[train_indices], y[train_indices], X[cv_indices], y[cv_indices]

def load_emnist(train_img, train_lbl, test_img, test_lbl):
    X_train = load_images(train_img)
    X_train = normalize(X_train)
    y_train = load_labels(train_lbl)
    y_train = one_hot(y_train)

    X_test = load_images(test_img)
    X_test = normalize(X_test)
    y_test = load_labels(test_lbl)
    y_test = one_hot(y_test)

    X_train, y_train, X_cv, y_cv = train_cv_split(X_train, y_train)

    return X_train, y_train, X_cv, y_cv, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_cv, y_cv, X_test, y_test = load_emnist(
        "../data/raw/emnist-letters-train-images-idx3-ubyte.gz",
        "../data/raw/emnist-letters-train-labels-idx1-ubyte.gz",
        "../data/raw/emnist-letters-test-images-idx3-ubyte.gz",
        "../data/raw/emnist-letters-test-labels-idx1-ubyte.gz",
    )

    print("Train images :", X_train.shape)
    print("Train labels :", y_train.shape)
    print("C.V images :", X_cv.shape)
    print("C.V labels :", y_cv.shape)
    print("Test images  :", X_test.shape)
    print("Test labels  :", y_test.shape)
