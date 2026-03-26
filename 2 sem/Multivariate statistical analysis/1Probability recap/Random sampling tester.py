
import numpy as np

# 1. Dine data (f.eks. 100 observationer af 3 variabler)
X = np.random.rand(100, 3)

# 2. Den "manuelle" matrix-måde (god for forståelsen)
X_mean = X - np.mean(X, axis=0)
cov_matrix_manual = (X_mean.T @ X_mean) / (X.shape[0] - 1)

# 3. Den lynhurtige måde (standard i industrien)
cov_matrix = np.cov(X, rowvar=False)

print(cov_matrix)