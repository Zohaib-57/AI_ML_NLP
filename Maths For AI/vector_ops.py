import numpy as np
from numpy.linalg import norm
import sympy

list1 = [1, 2, 3, 4, 5]
list2 = [6, 7, 8, 9, 10]

vector1 = np.array(list1)
vector2 = np.array(list2)
print("Horizontal Vector")
print(vector1)
print("Vertical Vector")
print(vector2)

new_vector1 = [2, 5, 3]
new_vector2 = [1, 4, 6]

# Operation on Vectors
addition = vector1 + vector2
print("Addition of two vectors : " + str(addition))

subtraction = vector1 - vector2
print("Subtraction of two vectors : " + str(subtraction))

multiplication = vector1 * vector2
print("Multiplication of two vectors : " + str(multiplication))

scalar = 2
print("Scalar : " + str(scalar))
scalar_mul = vector1 * scalar
print("Multiplication of vector with scalar : " + str(scalar_mul))

division = vector1 / scalar
print("Division of vector with scalar : " + str(division))

# Dot Product
dot_product = vector1.dot(vector2)
print("Dot Product of two vectors : " + str(dot_product))

# Cross Product
cross_product = np.cross(new_vector1, new_vector2)
print("Cross Product of two vectors : " + str(cross_product))

# Vector Norms
arr = np.array([1, 3, 5])
l1 = norm(arr, 1)
print("L1 Norm : " + str(l1))

arr2 = np.array([1, 3, 5])
l2 = norm(arr2, 2)
print("L2 Norm : " + str(l2))

arr3 = np.array([1, 3, 5])
vecMax = norm(arr3, np.inf)
print("L∞ Norm : " + str(vecMax))

# Projection Of Vector
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

v_norm = np.sqrt(sum(v**2))
proj_of_u_on_v = (np.dot(u, v) / v_norm**2) * v
print("Projection of u on v : " + str(proj_of_u_on_v))

# Linear Combination of Vectors
x = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0]]
)
y = ([3.65, 1.55, 3.42])
scalars = np.linalg.solve(x, y)
print(scalars)

# Linearly dependent and Independent Vecots
matrix = np.array([
    [0, 5, 1],
    [0, 10, 0]
])
_, indexes = sympy.Matrix(matrix).T.rref()
print("Linearly Independent Vectors : " + str(indexes))
print(matrix[indexes, :])

if len(indexes) == 2:
    print("Linealry Independent")
else:
    print("Linealry Dependent")
