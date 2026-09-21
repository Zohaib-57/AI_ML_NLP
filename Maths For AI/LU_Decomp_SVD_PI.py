import numpy as np
import scipy.linalg as la
from scipy.linalg import svd as scipy_svd
import matplotlib.pyplot as plt


np.set_printoptions(suppress=True)

A = np.array([
    [1, 3, 4],
    [2, 1, 3],
    [4, 1, 2]
])

L = np.array([
    [1, 0, 0],
    [2, 1, 0],
    [4, 11/5, 1]
])

U = np.array([
    [1, 3, 4],
    [0, -5, -5],
    [0, 0, -3]
])

print(L.dot(U))
print(L)
print(U)

# Moore_Penrose Pseudo_Inverse
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
print(X)
U, s, VT = scipy_svd(X)
print(U)
print(s)
print(VT)

# SVD

Z = np.array([
    [7, 2],
    [3, 4],
    [5, 3]
])

print(Z)
print(np.linalg.pinv(Z))
z1 = np.linspace(-5, 5, 1000)
z2_1 = -2*z1 + 2
z2_2 = 4*z1 + 8
z2_3 = -1*z1 + 2

plt.plot(z1, z2_1)
plt.plot(z1, z2_2)
plt.plot(z1, z2_3)
plt.xlim(-2, 1)
plt.ylim(1, 5)
plt.show()

# pinv used to compute the (Moore_penrose) pseudo-inverse of a matrix
Y = np.array([
    [-2, -1],
    [4, -1],
    [-1, -1]
])
Y_plus = np.linalg.pinv(A)
print("\n")
print(Y_plus)
