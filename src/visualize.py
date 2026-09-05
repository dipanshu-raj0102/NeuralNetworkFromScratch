import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, save_dir="../images"):
    """
    Saves:
        training_loss.png
        training_accuracy.png
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # -------- Loss --------
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], linewidth=2, label="Train")
    plt.plot(epochs, history["val_loss"], "--", linewidth=2, label="Validation")
    plt.title("Cross-Entropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_loss.png"), dpi=300)
    plt.close()

    # -------- Accuracy --------
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_acc"], linewidth=2, label="Train")
    plt.plot(epochs, history["val_acc"], "--", linewidth=2, label="Validation")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_accuracy.png"), dpi=300)
    plt.close()


def plot_confusion_matrix(model, X_test, y_test, save_dir="../images"):
    os.makedirs(save_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)

    num_classes = 26
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    cm_norm = cm.astype(np.float32)
    cm_norm /= cm_norm.sum(axis=1, keepdims=True)

    letters = [chr(ord("A") + i) for i in range(26)]

    plt.figure(figsize=(12, 10))
    plt.imshow(cm_norm, cmap="Blues")
    plt.colorbar(fraction=0.046)

    plt.xticks(range(26), letters)
    plt.yticks(range(26), letters)

    plt.xlabel("Predicted Letter")
    plt.ylabel("True Letter")
    plt.title("Normalized Confusion Matrix")

    # Numbers inside cells
    for i in range(num_classes):
        for j in range(num_classes):
            value = cm_norm[i, j]
            if value > 0.005:
                color = "white" if value > 0.5 else "black"
                plt.text(
                    j, i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=color
                )

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


def plot_predictions(model, X_test, y_test, save_dir="../images", n=36):
    os.makedirs(save_dir, exist_ok=True)

    idx = np.random.choice(len(X_test), n, replace=False)

    probs = model.forward(X_test[idx])
    preds = np.argmax(probs, axis=1)
    labels = np.argmax(y_test[idx], axis=1)

    plt.figure(figsize=(12, 12))

    for i in range(n):
        plt.subplot(6, 6, i + 1)

        plt.imshow(X_test[idx[i]].reshape(28, 28), cmap="gray")

        pred = chr(preds[i] + 65)
        true = chr(labels[i] + 65)
        conf = probs[i, preds[i]] * 100

        color = "green" if preds[i] == labels[i] else "red"

        plt.title(
            f"{pred}\n{conf:.0f}%",
            fontsize=8,
            color=color,
        )
        plt.xlabel(f"T:{true}", fontsize=7)
        plt.xticks([])
        plt.yticks([])

    plt.suptitle("Random Test Predictions", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "predictions.png"), dpi=300)
    plt.close()    

def plot_layer1_weights(model, save_dir="../images", n=16):
    """
    Visualizes the learned weights of the first Dense layer.
    Each neuron (784 weights) is reshaped into 28x28.
    """
    os.makedirs(save_dir, exist_ok=True)

    weights = model.layer1.W.T  # (128, 784)

    plt.figure(figsize=(8, 8))

    for i in range(n):
        plt.subplot(4, 4, i + 1)

        image = weights[i].reshape(28, 28)

        plt.imshow(image, cmap="seismic")
        plt.title(f"Neuron {i+1}", fontsize=8)
        plt.axis("off")

    plt.suptitle("Learned Features - Layer 1", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "layer1_weights.png"), dpi=300)
    plt.close()


def plot_misclassified(model, X_test, y_test, save_dir="../images", n=36):
    os.makedirs(save_dir, exist_ok=True)

    probs = model.forward(X_test)
    preds = np.argmax(probs, axis=1)
    labels = np.argmax(y_test, axis=1)

    wrong = np.where(preds != labels)[0]

    np.random.shuffle(wrong)
    wrong = wrong[:n]

    plt.figure(figsize=(12, 12))

    for i, idx in enumerate(wrong):
        plt.subplot(6, 6, i + 1)

        plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")

        pred = chr(preds[idx] + 65)
        true = chr(labels[idx] + 65)
        conf = probs[idx, preds[idx]] * 100

        plt.title(
            f"{pred}\n{conf:.0f}%",
            fontsize=8,
            color="red",
        )
        plt.xlabel(f"T:{true}", fontsize=7)
        plt.xticks([])
        plt.yticks([])

    plt.suptitle("Misclassified Test Samples", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "misclassified.png"), dpi=300)
    plt.close()
