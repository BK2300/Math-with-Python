import numpy as np
import matplotlib.pyplot as plt

# Enhedsbasis
e1 = np.array([1, 0])
e2 = np.array([0, 1])

# En matrix, som roterer og skalerer
A = np.array([[0, -1],
              [1,  0]])  # 90 graders rotation

# Transformér basisvektorerne
Ae1 = A @ e1
Ae2 = A @ e2

# Plot originalt og transformeret koordinatsystem
plt.quiver(0, 0, e1[0], e1[1], color='gray', scale=1, scale_units='xy', angles='xy', label='e1')
plt.quiver(0, 0, e2[0], e2[1], color='gray', scale=1, scale_units='xy', angles='xy', label='e2')
plt.quiver(0, 0, Ae1[0], Ae1[1], color='blue', scale=1, scale_units='xy', angles='xy', label='A·e1')
plt.quiver(0, 0, Ae2[0], Ae2[1], color='red', scale=1, scale_units='xy', angles='xy', label='A·e2')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.grid()
plt.legend()
plt.title("Matrix som lineær transformation")
plt.show()
