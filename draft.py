import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy import ndimage
from scipy import misc
from scipy.ndimage import convolve1d

# n=5
# sigma=1
#
# x = np.arange(-1*n, n+1)
# print(x)
#
# filter = np.exp((x**2)/-2/(sigma**2))
# filter = filter[1:-1]
# for i in filter:
#     print(i)
#
# print(type(filter))

img = np.array([[12,16,18],[13,11,30],[15,15,16]])

filter = np.array([1,1,1])

all_values_1 = np.array([[1,1,1],[1,1,1],[1,1,1]])

weight = convolve1d(all_values_1,filter,1,np.float64,'constant')
result = convolve1d(img,filter,1,np.float64,'constant')
norm_result = result/weight

print(weight)
print(result)
print(norm_result)
