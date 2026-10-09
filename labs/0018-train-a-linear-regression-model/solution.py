import numpy as np


def train(X, y, W, b, learning_rate=0.1, epochs=2000, tol=1e-10):
    """
    Train a linear regression model (y_hat = X @ W + b) with batch gradient descent
    on the mean squared error.

    Parameters
    ----------
    X : ndarray (n_samples, n_features), standardized training features
    y : ndarray (n_samples,) or (n_samples, 1), training targets
    W : ndarray, randomly initialized weights, shape (n_features,) or (n_features, 1)
    b : float or ndarray, initial bias
    learning_rate, epochs, tol : gradient descent settings

    Returns
    -------
    (W, b) : trained weights and bias, with the same shapes as the inputs
    """
    X = np.asarray(X, dtype=float)
    W = np.array(W, dtype=float)          # copies, so the inputs are not modified
    b = np.array(b, dtype=float)
    n = X.shape[0]
    y = np.asarray(y, dtype=float).reshape((X @ W).shape)  # match prediction shape

    for _ in range(epochs):
        error = X @ W + b - y             # prediction error, shape like X @ W
        grad_W = (X.T @ error) * (2.0 / n)
        grad_b = 2.0 * error.mean(axis=0) if error.ndim > 1 else 2.0 * error.mean()

        W -= learning_rate * grad_W
        b = b - learning_rate * grad_b

        if np.linalg.norm(grad_W) < tol and np.all(np.abs(grad_b) < tol):
            break

    return W, (float(b) if b.ndim == 0 else b)