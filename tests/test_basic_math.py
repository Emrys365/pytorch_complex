import operator

import numpy
import pytest
import torch

from torch_complex.tensor import ComplexTensor


def _get_complex_array(*shape):
    return numpy.random.randn(*shape) + 1j + numpy.random.randn(*shape)


@pytest.mark.parametrize('op',
                         [operator.add,
                          lambda x, y: x.__radd__(y),
                          operator.iadd,
                          operator.sub,
                          lambda x, y: x.__rsub__(y),
                          operator.isub,
                          operator.mul,
                          lambda x, y: x.__rmul__(y),
                          operator.imul,
                          operator.truediv,
                          lambda x, y: x.__rtruediv__(y),
                          operator.itruediv,
                          operator.matmul,
                          lambda x, y: x.__rmatmul__(y),
                          ])
@pytest.mark.parametrize('one_is_real', [True, False])
def test_binary_operation(op, one_is_real):
    n1 = _get_complex_array(10, 10)
    if one_is_real:
        n2 = numpy.random.randn(10, 10)
    else:
        n2 = _get_complex_array(10, 10)
    t1 = ComplexTensor(n1.copy())
    t2 = ComplexTensor(n2.copy())

    x = op(n1, n2)
    y = op(t1, t2)
    y = y.numpy()
    numpy.testing.assert_allclose(x, y)


@pytest.mark.parametrize('nop,top',
                         [(numpy.linalg.inv, lambda x: x.inverse()),
                          (lambda x: x.conj(), None),
                          (operator.neg, None)])
def test_unary_operation(nop, top):
    if top is None:
        top = nop
    n1 = _get_complex_array(10, 10)
    t1 = ComplexTensor(n1.copy())

    x = nop(n1)
    y = top(t1)
    y = y.numpy()
    numpy.testing.assert_allclose(x, y)


@pytest.mark.parametrize('num', [-2, -1, 0, 1, 2, 0.5, -0.5, 1.5, -1.5])
def test_pow(num):
    n1 = _get_complex_array(10, 10)
    t1 = ComplexTensor(n1.copy())

    x = n1 ** num
    y = t1 ** num
    y = y.numpy()
    numpy.testing.assert_allclose(x, y)


def test_sum():
    n1 = _get_complex_array(10, 10)
    t1 = ComplexTensor(n1.copy())
    x = n1.sum()
    y = t1.sum()
    y = y.numpy()
    numpy.testing.assert_allclose(x, y)


def test_mean():
    n1 = _get_complex_array(10, 10)
    t1 = ComplexTensor(n1.copy())
    x = n1.mean()
    y = t1.mean()
    y = y.numpy()
    numpy.testing.assert_allclose(x, y)


@pytest.mark.parametrize('shape', [(2, 2), (4, 4), (8, 8), (2, 4, 4), (10, 5, 5), (100, 10, 10), (1000, 20, 20)])
def test_symeig(shape):
    Z = ComplexTensor(torch.rand(*shape), torch.rand(*shape))
    Z = Z + Z.conj().transpose(-1, -2)
    eigvals_torch, eigvecs_torch = Z.symeig(eigenvectors=True)
    eigvals_numpy, eigvecs_numpy = numpy.linalg.eigh(Z.numpy())

    # Check if eigenvalues calculated by numpy and torch_complex are close to each other.
    # However, the eigenvectors calculated by these two methods are not necessarily similar,
    #   due to arbitrary phase rotations (see http://numpy-discussion.10968.n7.nabble.com/One-question-about-the-numpy-linalg-eig-routine-td5392.html)
    numpy.testing.assert_allclose(eigvals_torch.numpy(), eigvals_numpy, atol=1e-5)
    # Validate (eigenvalue, eigenvector) pairs
    numpy.testing.assert_allclose(numpy.matmul(Z.numpy(), eigvecs_torch.numpy()),
                                  eigvecs_torch.numpy() * eigvals_torch.unsqueeze(-2).numpy(), atol=1e-5)