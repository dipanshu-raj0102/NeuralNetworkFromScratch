
import matplotlib.pyplot as plt
import numpy as np

from data_loader import (
    load_emnist,
)
from model import *
from optimizer import SGDOptimizer


def evaluate(model, X, y_one_hot):
    y_pred_probs = model.forward(X)
    loss = model.compute_loss(y_one_hot)

    predictions = np.argmax(y_pred_probs, axis=1)
    targets = np.argmax(y_one_hot, axis=1)
    accuracy = np.mean(predictions == targets) * 100.0

    return loss, accuracy


def fit(
    model, optimizer, X_train, y_train, X_val, y_val, epochs=15, batch_size=128
):
    num_samples = X_train.shape[0]

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    print(
        f"\n--- Starting Training ({epochs} Epochs, Batch Size: {batch_size}) ---"
    )

    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(num_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            model.forward(X_batch)

            model.compute_loss(y_batch)

            model.backward(y_batch)

            optimizer.step()

        train_loss, train_acc = evaluate(model, X_train, y_train)
        val_loss, val_acc = evaluate(model, X_val, y_val)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

    return history


def plot_training_history(history):
    """Plots Loss and Accuracy curves over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss Plot
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linestyle="--")
    axes[0].set_title("Cross-Entropy Loss vs. Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy Plot
    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", linestyle="--")
    axes[1].set_title("Accuracy vs. Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
      
    print("Loading datasets...")
    X_train, y_train, X_cv, y_cv, X_test, y_test = load_emnist(
        "../data/raw/emnist-letters-train-images-idx3-ubyte.gz",
        "../data/raw/emnist-letters-train-labels-idx1-ubyte.gz",
        "../data/raw/emnist-letters-test-images-idx3-ubyte.gz",
        "../data/raw/emnist-letters-test-labels-idx1-ubyte.gz",
    )


    model = MLP(input_dim=784, num_classes=26)

    trainable_layers = [model.layer1, model.layer2, model.layer3]
    optimizer = SGDOptimizer(layers=trainable_layers, lr=2)

    history = fit(
        model=model,
        optimizer=optimizer,
        X_train=X_train,
        y_train=y_train,
        X_val=X_cv,
        y_val=y_cv,
        epochs=20,
        batch_size=128,
    )

    test_loss, test_acc = evaluate(model, X_test, y_test)
    print("\n--- Final Test Results ---")
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

    plot_training_history(history)
