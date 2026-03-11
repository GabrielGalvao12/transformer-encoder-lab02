import numpy as np


class FeedForward:

    def __init__(self, d_model, d_ff=128):

        self.W1 = np.random.randn(d_model, d_ff)
        self.b1 = np.zeros((d_ff))

        self.W2 = np.random.randn(d_ff, d_model)
        self.b2 = np.zeros((d_model))

    def forward(self, X):

        hidden = X @ self.W1 + self.b1

        hidden = np.maximum(0, hidden)  # ReLU

        output = hidden @ self.W2 + self.b2

        return output