import sys
import numpy as np
from numpy import pi
from numpy import newaxis
rg = np.random.default_rng(1)  # create instance of default random number generator

########################################################################################################################
# Changing the shape of an array
print("\nChanging the shape\n")

"An array has a shape given by the number of elements along each axis:"

a = np.floor(10 * rg.random((3, 4)))  # Create array xx4 of random floats
print(a)
"""array([[3., 7., 3., 4.],
       [1., 4., 2., 2.],
       [7., 2., 4., 9.]])"""
print(a.shape)  # (3, 4)

"""The shape of an array can be changed with various commands. 
Note that the following three commands all return a modified array, but do not change the original array:"""

print(a.ravel())  # returns the array, flattened
# array([3., 7., 3., 4., 1., 4., 2., 2., 7., 2., 4., 9.])
print(a.reshape(6, 2))  # returns the array with a modified shape
"""array([[3., 7.],
       [3., 4.],
       [1., 4.],
       [2., 2.],
       [7., 2.],
       [4., 9.]])"""
print(a.T)  # returns the array, transposed
"""array([[3., 1., 7.],
       [7., 4., 2.],
       [3., 2., 4.],
       [4., 2., 9.]])"""
print(a.T.shape)  # (4, 3)
print(a.shape)  # (3, 4)

"""The reshape function returns its argument with a modified shape, 
whereas the ndarray.resize method modifies the array itself:"""

print(a)  # original array
"""array([[3., 7., 3., 4.],
       [1., 4., 2., 2.],
       [7., 2., 4., 9.]])"""
a.resize((2, 6))  # change the array itself
print(a)
"""array([[3., 7., 3., 4., 1., 4.],
       [2., 2., 7., 2., 4., 9.]])"""

"""If a dimension is given as -1 in a reshaping operation, the other dimensions are automatically calculated:"""

a.reshape(3, -1)  # reshape with unknown last parameter -> auto calculate
"""array([[3., 7., 3., 4.],
       [1., 4., 2., 2.],
       [7., 2., 4., 9.]])"""

########################################################################################################################
print("\nStacking together different arrays\n")

"""Several arrays can be stacked together along different axes:"""

a = np.floor(10 * rg.random((2, 2)))  # new array 2x2
print(a)
"""array([[9., 7.],
       [5., 2.]])"""
b = np.floor(10 * rg.random((2, 2)))  # new array 2x2
print(b)
"""array([[1., 9.],
       [5., 1.]])"""
print(np.vstack((a, b)))  # stack vertically
"""array([[9., 7.],
       [5., 2.],
       [1., 9.],
       [5., 1.]])"""
print(np.hstack((a, b)))  # Stack horizontally
"""array([[9., 7., 1., 9.],
       [5., 2., 5., 1.]])"""

"""The function column_stack stacks 1D arrays as columns into a 2D array. 
It is equivalent to hstack only for 2D arrays:"""

print(np.column_stack((a, b)))  # with 2D arrays
"""array([[9., 7., 1., 9.],
       [5., 2., 5., 1.]])"""
a = np.array([4., 2.])
b = np.array([3., 8.])
print(np.column_stack((a, b)))  # returns a 2D array
"""array([[4., 3.],
       [2., 8.]])"""
print(np.hstack((a, b)))        # the result is different
"""array([4., 2., 3., 8.])"""
print(a[:, newaxis])  # view `a` as a 2D column vector
"""array([[4.],
       [2.]])"""
print(np.column_stack((a[:, newaxis], b[:, newaxis])))  # stacks columns
"""array([[4., 3.],
       [2., 8.]])"""
print(np.hstack((a[:, newaxis], b[:, newaxis])))  # the result is the same
"""array([[4., 3.],
       [2., 8.]])"""

"""On the other hand, the function row_stack is equivalent to vstack for any input arrays. 
In fact, row_stack is an alias for vstack:

In general, for arrays with more than two dimensions, 
    hstack stacks along their second axes, 
    vstack stacks along their first axes, 
    and concatenate allows for an optional arguments giving the number of the axis along which the concatenation 
    should happen."""


print(np.column_stack is np.hstack)  # False
print(np.row_stack is np.vstack)  # True

"""In complex cases, r_ and c_ are useful for creating arrays by stacking numbers along one axis. 
They allow the use of range literals :."""

print(np.r_[1:4, 0, 4])  # create complex array
"""array([1, 2, 3, 0, 4])"""

########################################################################################################################
print("\nSplitting one array into several smaller ones\n")

"""Using hsplit, you can split an array along its horizontal axis, 
either by specifying the number of equally shaped arrays to return, 
or by specifying the columns after which the division should occur:

vsplit splits along the vertical axis, and array_split allows one to specify along which axis to split.
"""

a = np.floor(10 * rg.random((2, 12)))
print(a)
"""array([[6., 7., 6., 9., 0., 5., 4., 0., 6., 8., 5., 2.],
       [8., 5., 5., 7., 1., 8., 6., 7., 1., 8., 1., 0.]])"""
# Split `a` into 3
print(np.hsplit(a, 3))
"""[array([[6., 7., 6., 9.],
       [8., 5., 5., 7.]]), 
 array([[0., 5., 4., 0.],
       [1., 8., 6., 7.]]), 
 array([[6., 8., 5., 2.],
       [1., 8., 1., 0.]])]"""
# Split `a` after the third and the fourth column
print(np.hsplit(a, (3, 4)))
"""[array([[6., 7., 6.],
       [8., 5., 5.]]), 
 array([[9.],
       [7.]]), 
 array([[0., 5., 4., 0., 6., 8., 5., 2.],
       [1., 8., 6., 7., 1., 8., 1., 0.]])]"""
