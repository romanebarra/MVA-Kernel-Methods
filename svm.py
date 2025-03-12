import numpy as np
import cvxopt
from cvxopt import matrix, solvers

# kernel SVM solved with cvxopt
def kernel_svm(K, y, C):
    """
    Parameters:
        K : kernel Gram matrix
        y : labels
        C : regularization parameter
    Returns:
        alpha : dual variable
        b : bias term
    """
    n = len(y)
    Q = np.outer(y, y) * K # matrix of the dual problem  Q = (y_i * y_j * K_ij)_i,j
    Q = matrix(Q)

    q = matrix(-np.ones(n)) # objective function linear term min - sum alpha_i

    # inequality constraint Gx <= h
    # constraint -alpha_i <= 0 and alpha_i <= C
    G = np.vstack((-np.eye(n), np.eye(n))) 
    G = matrix(G)
    h = np.hstack((np.zeros(n), C * np.ones(n))) 
    h = matrix(h)

    # equality constraint Ax = b
    # constraint sum (alpha_i * y_i) = 0
    A = matrix(y, (1, n), tc='d')
    b = matrix(0.0)

    sol = cvxopt.solvers.qp(Q, q, G, h, A, b) # solves the quadratic problem
    alpha = np.ravel(sol['x'])

    tol = 1e-5
    support_vectors = np.where((alpha > tol) & (alpha < C - tol))[0] # support vectors

    if len(support_vectors) > 0:
        bias = np.mean([y[i] - np.sum(alpha * y * K[i, :]) for i in support_vectors]) # average of the bias term
    else:
        bias = 0.0
    return alpha, bias
    

# Comupute the prediction of the labels given the output of the svm
def predict_svm(K_test_train, y_train, alpha, bias):
    f = np.dot(K_test_train, alpha * y_train) + bias
    return np.sign(f)