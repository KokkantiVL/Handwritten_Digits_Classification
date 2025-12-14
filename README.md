# Handwritten Digits Classification

This project is a programming assignment for Machine Learning. The goal is to implement a Multilayer Perceptron (MLP) Neural Network to classify handwritten digits from the MNIST dataset. The model is trained using feedforward and backpropagation techniques with L2 regularization and hyperparameter tuning.

---

##  Project Objectives

- Understand and implement the inner workings of a feedforward neural network.
- Apply backpropagation to compute gradients and train the model.
- Incorporate regularization to control model complexity.
- Tune hyperparameters using a validation set to balance bias and variance.
- Evaluate the trained model’s accuracy on the test set.

---

##  Files

- `nnScript.py`: Main script containing function definitions:
  - `preprocess()`: Loads and preprocesses the MNIST data.
  - `sigmoid()`: Computes the sigmoid activation.
  - `nnObjFunction()`: Computes the objective function and gradients with regularization.
  - `nnPredict()`: Predicts class labels using trained weights.
  - `initializeWeights()`: Randomly initializes weights.
- `mnist_all.mat`: The dataset containing digit images from 0 to 9, split into train and test matrices.
- `params.pickle`: Pickled object storing the learned parameters:
  - `w1`: Weights between input and hidden layer.
  - `w2`: Weights between hidden and output layer.
  - `optimal_n_hidden`: Chosen number of hidden units.
  - `optimal_lambda`: Chosen regularization parameter.
- `README.md`: This file.
- `report/`: Folder containing the report (`.pdf`) explaining the experiment, results, and hyperparameter tuning.

---

##  How to Run

1. **Environment Setup**

   Ensure the following Python packages are installed:

   ```bash
   pip install numpy scipy matplotlib scikit-learn

2. Launch the script in a Python environment:
   ```bash
   python nnScript.py

3.	Model Training and Evaluation
The script performs the following:
	•	Preprocesses and normalizes the MNIST dataset.
	•	Trains the neural network using scipy.optimize.minimize with conjugate gradient descent.
	•	Tunes hyperparameters (λ and number of hidden units).
	•	Evaluates model performance on training, validation, and test sets.
