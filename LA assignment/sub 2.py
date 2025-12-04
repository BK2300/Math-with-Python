import numpy as np

A = np.array([[8, 0, 0],
              [0, 6, 0],
              [2, 1, 3]], dtype=float)

def make_b(d1, d2, d3, d4):
    c2, c3, c4 = 64.0, 36.0, 14.0
    return 0.5 * np.array([
        d1**2 - d2**2 + c2,
        d1**2 - d3**2 + c3,
        d1**2 - d4**2 + c4
    ])

distances = [
    (3.74, 5.48, 5.10, 2.45),
    (13.55, 8.92, 10.84, 11.51),
    (23.45, 18.06, 20.15, 21.35)
]

for k, (d1, d2, d3, d4) in enumerate(distances, start=1):
    b = make_b(d1, d2, d3, d4)
    x = np.linalg.solve(A, b)
    print(f"k = {k}: b = {b}, x = {x}")
