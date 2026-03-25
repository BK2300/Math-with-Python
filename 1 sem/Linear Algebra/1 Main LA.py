import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ Lav nogle vektorer
v = np.array([2, 3])
w = np.array([1, -1])

print("v =", v)
print("w =", w)

# 2️⃣ Addition og skalering
print("v + w =", v + w)
print("2 * v =", 2 * v)

# 3️⃣ Dot product (indre produkt)
dot = np.dot(v, w)
print("Dot product v·w =", dot)

# 4️⃣ Visualisér vektorerne
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='blue', label='v')
plt.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, color='red', label='w')
plt.xlim(-2, 4)
plt.ylim(-2, 4)
plt.grid()
plt.legend()
plt.title("Visualisering af vektorer")
plt.show()





















