import numpy as np

i = np.array([[0,0,1],
              [1,1,1],
              [1,0,1],
              [0,1,1]])

w = np.array([[2],
              [1],
              [1]])

x = i*w

print(x)
