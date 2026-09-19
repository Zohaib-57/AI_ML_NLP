import numpy as np 

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

print("\nMatrix x:")
print(x)
print("\nMatrix y:")
print(y)

print("\nMatrix addition " + str(np.add(x, y)))
print("\nMatrix subtraction " + str(np.subtract(x, y)))
print("\nMatrix multiplication " + str(np.multiply(x, y)))

print("\nMatrix division " + str(np.divide(x, y)))
print("\nMatrix Square Root " + str(np.sqrt(x)))
print("\nThe summmation of all the matrix elements " + str(np.sum(y)))
print("\nThe column wise summation of all the matrix elements " + str(np.sum(y,axis =0)))
print("\nThe row wise summation of the all matrix is " + str(np.sum(x,axis =1)))
print("\nThe transpose of the matrix is " + str(np.transpose(x)))

a = np.array([1,2,3,4])
b = np.array([5,6,7,8])

print(("a= " , a))
print(("b= " , b))
print("\ninner:", np.inner(a,b))
print("dot:", np.dot(a,b))

print(a, type(a))
print(b,type(b))

print("\nMatrix multiplication of a * b : " + str(a * b))
print("\nMatrix mulitplication of b * a : " + str(b * a))
print("\nMatrix division " + str(a / b))