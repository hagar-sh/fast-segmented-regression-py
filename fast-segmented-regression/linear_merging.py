from collections import namedtuple
import numpy
import math

LinearPiece = namedtuple('LinearPiece', [
    'left_index',
    'right_index',
    'theta'
])


def fit_linear_piece (X: numpy.array, y: numpy.array, left_index: int, right_index: int):
    theta = numpy.linalg.solve(X[left_index:right_index, :], y[left_index:right_index])
    return LinearPiece(left_index=left_index, right_index=right_index, theta=theta)


def piece_length (p: LinearPiece):
    return p.right_index - p.left_index + 1


def linear_piece_merging_error(p: LinearPiece, X: numpy.array, y: numpy.array, sigma: float):
    return linear_piece_error(p, X, y) - piece_length(p) * sigma ^ 2


def linear_piece_error(p: LinearPiece, X: numpy.array, y: numpy.array):
    return numpy.linalg.norm(numpy.mat(y[p.left_index:p.right_index]) - numpy.mat(X[p.left_index:p.right_index, :]) * numpy.mat(p.theta)) ^ 2


def linear_fit_error(X: numpy.array, y: numpy.array, left_index: int, right_index: int):
    p = fit_linear_piece(X, y, left_index, right_index)
    return linear_piece_error(p, X, y)


def generate_equal_size_linear_data(endpoint_values, n, sigma):
    X = numpy.zeros([n, 2], dtype=float)
    X[:, 1] = numpy.ones(n, float)
    X[:, 2] = numpy.linspace(0, 1, n)

    num_segments = len(endpoint_values) - 1
    num_per_bin = math.floor(n / num_segments)
    num_bins_plusone = n % num_segments

    ystar = numpy.array([])
    for ii in range(1, num_bins_plusone):
        ystar = numpy.append(ystar, numpy.linspace(endpoint_values[ii]), endpoint_values[ii + 1], num_per_bin + 1)

    for ii in range((num_bins_plusone + 1), num_segments):
        ystar = numpy.append(ystar, numpy.linspace(endpoint_values[ii]), endpoint_values[ii + 1], num_per_bin)

    y = ystar + sigma * numpy.random.randn(n)
    return y, ystar, X


def generate_equal_size_random_regression_data(num_segments, n, d, sigma):
    X = numpy.random.randn(n, d)

    num_per_bin = math.floor(n / num_segments)
    num_bins_plusone = n % num_segments

    ystar = numpy.array([])
    cur_start = 1
    for ii in range(1, num_bins_plusone):
        cur_end = cur_start + num_per_bin
        beta = 2 * numpy.random.rand(d) + 1
        ystar = numpy.append(ystar, numpy.array(numpy.mat(X[cur_start:cur_end, :]) * beta).flatten())
        cur_start = cur_end + 1

    for ii in range((num_bins_plusone + 1), num_segments):
        cur_end = cur_start + num_per_bin - 1
        beta = 2 * numpy.random.rand(d) + 1
        ystar = numpy.append(ystar, numpy.array(numpy.mat(X[cur_start:cur_end, :]) * beta).flatten())
        cur_start = cur_end + 1

    y = ystar + sigma * numpy.random.randn(n)
    return y, ystar, X


def partition_to_vector(X: numpy.array, pieces):
    n = pieces[-1].right_index
    (rows, cols) = X.shape
    if n != rows:
        raise ValueError("number of rows and rightmost index must match")
    y = numpy.array([])
    for ii in range(1, len(pieces)):
        p = pieces[ii]
        y[p.left_index: p.right_index] = numpy.array(numpy.mat(X[p.left_index: p.right_index, :]) * numpy.mat(p.theta))
    return y


def mse(yhat, ystar):
    return (1.0 / len(yhat)) * sum((yhat - ystar) ^ 2)


def compute_errors_fast(X: numpy.array, y: numpy.array):
    (n, d) = X.shape
    res = -numpy.ones([n, n])
    for ii in range(1, n):
        normysq = 0.0
        A = numpy.matrix(numpy.zeros([d, d]))
        for jj in range(ii, min(ii + d - 1, n)):
            normysq += y[jj] * y[jj]
            A += numpy.mat(X[jj, :]) * numpy.mat(numpy.transpose(X[jj, :]))
            theta = numpy.linalg.solve(X[ii:jj, :], y[ii:jj])
            res[ii, jj] = normysq - float((numpy.mat(numpy.transpose(theta)) * A) * numpy.mat(theta))

        if ii + d - 1 < n:
            Ainv = numpy.invert(A)
            y2 = numpy.matrix(numpy.transpose(X[ii:ii + d - 1])) * numpy.mat(y[ii:ii+d-1])
