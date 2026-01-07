## Linear Regression from Scratch ##

# Overview
This repository contains my implementation of linear regression from scratch as part of WiDS midterm assignment.
The objective was to implement linear regression using:
- Closed-form solution
- Gradient descent optimisation

All computations are done manually using linear algebra.

# Dataset
- Input dataset: `dataset01.csv`
- The dataset is split into:
  - 80% training data
  - 20% testing data
- Features are real-valued, and the target variable is continuous.

# Implementation Details

## 1. Closed-form Solution
- Implemented the normal equation:
  \[
  w^* = (X^T X)^{-1} X^T y
  \]
- This computes the optimal weights directly using matrix inversion.

## 2. L2 Loss Function
- Implemented mean squared error (L2 loss):
  \[
  \mathcal{L}(w) = \frac{1}{n} \sum (y - Xw)^2
  \]

## 3. Gradient of L2 Loss
- Derived and implemented the gradient:
  \[
  \nabla_w \mathcal{L}(w) = -\frac{2}{n} X^T (y - Xw)
  \]

## 4. Gradient Descent Training
- Implemented gradient descent manually:
  - Iteratively updates weights
  - Stops when loss change falls below a small threshold
- Test loss is recorded at each iteration

---

# Results
- The model successfully converges using gradient descent
- Test loss decreases over iterations
- A plot of test error vs iterations is generated

---

# How to Run
1. Install dependencies:
   ```bash
   pip install numpy pandas torch matplotlib
