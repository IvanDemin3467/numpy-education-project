import sys
import numpy as np
from numpy import pi

a = np.arange(15).reshape(3, 5)  # Create array 3x5 of integers from 0 to 14
print(a)  # It is a 2-d array
print(a.shape)  # A tuple of the dimensions of the array:  (3, 5)
print(a.ndim)  # An integer representing the number of dimensions: 2
print(a.dtype.name)  # A string representing the name of the variables in the array: int32
print(a.itemsize)  # An integer = 32 / 8 = 4
print(a.size)  # An integer representing the size of the array: 15
print(type(a))  # A string representing the class of the array: "<class 'numpy.ndarray'>"

# Creation of the array
print("\narray\n")

print(np.array([2, 3, 4]))  # Create numpy array of int64 from Python list
print(np.array([1.2, 3.5, 5.1]))  # Array of float64
print(np.array([(1.5, 2, 3), (4, 5, 6)]))  # 2-d array from sequence of sequences
print(np.array([[1, 2], [3, 4]], dtype=complex))  # Explicitly specify a type of the array during creation (complex)
print(np.zeros((3, 4)))  # Create an array 3x4 filled with zeroes
print(np.ones((2, 3, 4), dtype=np.int16))  # Create an array of int16 2x3x4 filled with ones
print(np.empty((2, 3)))  # Create an array 2x3 filled with random data

# Creation of ranges with arange
print("\narange\n")

print(np.arange(10, 30, 5))  # Same as range(10, 30, 5) in Python, byt returns array
print(np.arange(0, 2, 0.3))  # It accepts float arguments

# Creation of ranges with linspace
print("\nlinspace\n")

print(np.linspace(0, 2, 9))  # 9 numbers from 0 to 2
x = np.linspace(0, 2 * pi, 100)  # Useful to evaluate function at lots of points
print(np.sin(x))

# Printing arrays
print("\nPrinting arrays\n")

print(np.arange(6))  # 1d array looks as a row
print(np.arange(12).reshape(4, 3))  # 2d array looks as a matrix
print(np.arange(24).reshape(2, 3, 4))  # 3d array looks as a list of matrices

print(np.arange(10000))  # NumPy automatically skips the central part of the array and only prints the corners
print(np.arange(10000).reshape(100, 100))

# Basic Operations
print("\nBasic Operations\n")

a = np.array([20, 30, 40, 50])
b = np.arange(4)
print(b)  # array([0, 1, 2, 3])
print(a - b)  # array([20, 29, 38, 47])
print(b**2)  # array([0, 1, 4, 9])
print(10 * np.sin(a))  # array([ 9.12945251, -9.88031624,  7.4511316 , -2.62374854])
print(a < 35)  # array([ True,  True, False, False])

# Matrices multiplication
print("\nMatrices multiplication\n")

A = np.array([[1, 1],
              [0, 1]])
B = np.array([[2, 0],
              [3, 4]])
print(A * B)     # elementwise product: array([[2, 0], [0, 4]])
print(A @ B)     # matrix product: array([[5, 4], [3, 4]])
print(A.dot(B))  # another matrix product: array([[5, 4], [3, 4]])

# +=

print("\n+=\n")
rg = np.random.default_rng(1)  # create instance of default random number generator
a = np.ones((2, 3), dtype=int)
b = rg.random((2, 3))
a *= 3
print(a)  # array([[3, 3, 3],  [3, 3, 3]])
b += a
print(b)  # array([[3.51182162, 3.9504637 , 3.14415961],  [3.94864945, 3.31183145, 3.42332645]])
try:
    a += b  # b is not automatically converted to integer type
except Exception as e:
    print(e)

# When operating with arrays of different types, the type of the resulting array corresponds
# to the more general or precise one (a behavior known as upcasting).

print("\nUpcasting\n")
a = np.ones(3, dtype=np.int32)
b = np.linspace(0, pi, 3)
print(b.dtype.name)  # 'float64'
c = a + b  # Trying to sum int with float
print(c)  # array([1.        , 2.57079633, 4.14159265])
print(c.dtype.name)  # 'float64'
d = np.exp(c * 1j)  # Creating array of complexes
print(d)  # array([ 0.54030231+0.84147098j, -0.84147098+0.54030231j, -0.54030231-0.84147098j])
print(d.dtype.name)  # 'complex128'

# Unary operators
print("\nUnary operators\n")

a = rg.random((2, 3))
print(a)  # array([[0.82770259, 0.40919914, 0.54959369], [0.02755911, 0.75351311, 0.53814331]])
print(a.sum())  # 3.1057109529998157
print(a.min())  # 0.027559113243068367
print(a.max())  # 0.8277025938204418

# By default, these operations apply to the array as though it were a list of numbers, regardless of its shape.
# However, by specifying the axis parameter you can apply an operation along the specified axis of an array:
print("\nAxis operators\n")

b = np.arange(12).reshape(3, 4)
print(b)
"""array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])"""
print(b.sum(axis=0))     # sum of each column: array([12, 15, 18, 21])
print(b.min(axis=1))     # min of each row: array([0, 4, 8])
print(b.cumsum(axis=1))  # cumulative sum along each row
"""array([[ 0,  1,  3,  6],
       [ 4,  9, 15, 22],
       [ 8, 17, 27, 38]])"""
