"""
Linear Regression model
"""

import numpy as np


class Linear(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # Initialize in train
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.weight_decay = weight_decay

    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the linear regression update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        def encoding(labels, num_classes):
            return np.eye(num_classes)[labels]

        encoded_y = encoding(y_train, self.n_class)

        N, D = X_train.shape
        self.w = weights
        # TODO: implement me
        for e in range(self.epochs):
            y_pred = np.dot(X_train, self.w.T) 
            gradient = (2/N) * np.dot((y_pred - encoded_y).T, X_train)
            self.w -= self.lr * gradient
            self.w *= (1 - self.lr * self.weight_decay)


        return self.w

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        y_pred = np.dot(X_test, self.w.T) 
        return np.argmax(y_pred, axis=1)

