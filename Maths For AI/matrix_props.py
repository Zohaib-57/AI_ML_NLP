import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


x = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
y = ([3.56, 1.55, 3.42])

scalars = np.linalg.solve(x, y)
print(scalars)

a = np.arange(-10, 10)
b = 2 * a + 1

# plt.figure()
# plt.plot(a, b)
# plt.xlim(-2, 10)
# plt.ylim(-2, 10)

# plt.axvline(x=0, color='#A9A9A9', linewidth=1)
# plt.axhline(y=0, color='#A9A9A9', linewidth=1)
# plt.show()
# plt.close()


a = np.arange(-10, 10)
b = 2 * a + 1
c = 6 * a - 2

# plt.figure()
# plt.plot(a, b)
# plt.plot(a, c)
# plt.xlim(-2, 10)
# plt.ylim(-2, 10)

# plt.axvline(x=0, color='#A9A9A9', linewidth=1)
# plt.axhline(y=0, color='#A9A9A9', linewidth=1)
# plt.show()
# plt.close()

# Trace
x1 = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]])
print("x1 = ")
print(x1)
print("\nTrace:", x1.trace())
print("Trace:", sum(x1.diagonal()))

# Rank
a = np.arange(1, 10)
a = a.reshape(3, 3)
print("\nRank:", np.linalg.matrix_rank(a))

# Determinant
x2 = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [4, 5, 6]
])
print("x2 = ")
print(x2)
det = np.linalg.det(x2)
print("\nDeterminant:", det)

# True Inverse
x3 = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [4, 5, 6]
])
print("x3 = ")
print(x3)
det = np.linalg.det(x3)
print("\nDeterminant:", np.round(det))

if det != 0:
    inv = np.linalg.inv(x3)
    print("\nInverse:", inv)
else:
    print("\nMatrix is singular — no inverse exists.")
    pinv = np.linalg.pinv(x3)
    print("Pseudo-Inverse (approximate):", pinv)

a = np.arange(1, 10)
a.shape = (3, 3)
print("a = ")
print(a)
print("\nAfter Falttening")
print("________________________")
print(a.flatten())


# EigenValue nad EigenVectors
a = np.array ([
    [2,4,5],
    [3,5,6],
    [1,2,3]
    ])

print("a = ")
print(a)
w,v = np.linalg.eig(a)
print("\nEigenValues = :")
print(w)
print("\nEigenVectors = :")
print(v)
