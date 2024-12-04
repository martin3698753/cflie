import numpy as np

class QNU:
    def __init__(self):
        self.linear_weights = np.random.randn(9, 1)
        self.quadratic_weights = np.random.randn(9, 9, 1)
        self.bias = np.zeros((1, 1))

    def forward(self, X):
        linear_term = np.dot(X, self.linear_weights)
        quadratic_term = np.einsum('ij,jk,ik->ik', X, self.quadratic_weights, X)
        return linear_term + quadratic_term + self.bias

    def backward(self, X, dL_dout):
        # Calculate gradients
        dL_dlinear = dL_dout
        dL_dquadratic = dL_dout

        # Calculate gradients of linear and quadratic terms w.r.t. weights and biases
        dL_dw = np.dot(X.T, dL_dlinear)
        dL_dW = np.einsum('ij,ik,lm->jklm', X, X, dL_dquadratic)
        dL_db = np.sum(dL_dout, axis=0, keepdims=True)

        # Update weights and biases using gradient descent
        learning_rate = 0.01
        self.linear_weights -= learning_rate * dL_dw
        self.quadratic_weights -= learning_rate * dL_dW
        self.bias -= learning_rate * dL_db
