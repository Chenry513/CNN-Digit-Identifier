import numpy as np

def mlrloss(wb, X, y, K, gpu=0, prediction=0):
    if gpu == 1:
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float64)

    # N features, M examples
    # K distinct classes (1 to K)
    N, M = X.shape
    theta = wb[:N*(K-1)].reshape(K-1, N)
    bias = wb[N*(K-1):].reshape(K-1, 1)

    I = np.zeros((K, M))
    for i, yi in enumerate(y[0]):
        I[int(yi), i] = 1

    # Compute the values after the linear transform
    W = np.vstack([theta @ X + bias, np.zeros((1, M))])

    # Rescale to avoid overflow with exp operation
    W -= np.max(W, axis=0)
    W = np.exp(W)

    # Convert to probabilities by normalizing
    P = W / np.sum(W, axis=0)

    # Loss
    nll = -np.sum(np.log(P[I == 1]))
    if prediction == 1:
        indices = np.argmax(P, axis=0)
        percent = np.sum(y == indices) / len(y[0])
    else:
        percent = 0

    # Compute the gradients
    od = P - I
    gw = od @ X.T
    gw = gw[:K-1, :]
    gb = np.sum(od, axis=1).reshape(-1, 1)
    gb = gb[:K-1, :]
    g = np.concatenate([gw.ravel(), gb.ravel()])

    # Compute the derivatives for backprop
    od = theta.T @ od[:K-1, :]

    return nll, g, od, percent
