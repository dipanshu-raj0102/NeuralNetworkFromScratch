Neural Network (MLP) from Scratch for Handwritten Letter Recognition (NumPy)

Build a fully connected Multi-Layer Perceptron (MLP) from scratch using only NumPy to recognize handwritten English letters (A–Z). No TensorFlow, PyTorch, or high-level deep learning libraries.

Project Roadmap

Phase 1 — Project Setup

- Create repository structure
- Download & organize EMNIST Letters dataset
- Image preprocessing (normalize & flatten)
- One-hot encode labels

Phase 2 — Mathematical Foundations

- Matrix multiplication & tensor shapes
- Dense layer mathematics
- ReLU activation
- Softmax activation
- Cross-Entropy loss

Phase 3 — Forward Propagation

- Parameter initialization
- Implement Dense layer
- Build multi-layer forward pass
- Generate class probabilities

Phase 4 — Backpropagation

- Derive gradients manually
- Backprop through Softmax + Cross-Entropy
- ReLU derivative
- Update weights with Gradient Descent

Phase 5 — Training Engine

- Mini-batch gradient descent
- Learning rate tuning
- Epoch training loop
- Track loss & accuracy

Phase 6 — Evaluation

- Test accuracy
- Confusion matrix
- Misclassified letter analysis
- Prediction visualization

Phase 7 — Documentation

- Mathematical derivations
- Architecture diagrams
- Training visualizations
- Complete README
- LinkedIn project post

Target Architecture

"784 → 128 → 64 → 26"

- Input: 28×28 image (784 features)
- Hidden 1: 128 neurons + ReLU
- Hidden 2: 64 neurons + ReLU
- Output: 26 neurons + Softmax

Tech Stack

- Python 3
- NumPy
- Matplotlib
- Pandas (data handling)

Learning Goals

- Implement neural networks from first principles
- Understand forward & backward propagation
- Derive gradients without autograd
- Build a complete deep learning training pipeline using only NumPy

Progress Tracker

- [ ] Phase 1 — Project Setup
- [ ] Phase 2 — Mathematical Foundations
- [ ] Phase 3 — Forward Propagation
- [ ] Phase 4 — Backpropagation
- [ ] Phase 5 — Training Engine
- [ ] Phase 6 — Evaluation
- [ ] Phase 7 — Documentation
