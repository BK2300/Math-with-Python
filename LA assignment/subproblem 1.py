import numpy as np

# adjacency matrix
A = np.array([
    [0,1,1,0,0,0],
    [1,0,1,1,0,0],
    [1,1,0,0,1,0],
    [0,1,0,0,1,1],
    [0,0,1,1,0,1],
    [0,0,0,1,1,0]
], dtype=float)

beta = 0.05
gamma = 0.125
I = np.eye(6)

U = (1 - gamma) * I + beta * A

x0 = np.array([1,0,0,0,0,0], dtype=float)

def Tk(k):
    return np.linalg.matrix_power(U, k) @ x0

x1  = Tk(1)
x25 = Tk(25)
x50 = Tk(50)
print(x1, x25, x50)
