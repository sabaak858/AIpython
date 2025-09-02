import numpy as np
A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]], dtype=float)
A_inv = np.linalg.inv(A)
print("A inverse:\n", A_inv)
print("\nA * A_inv:\n", A @ A_inv)
print("\nA_inv * A:\n", A_inv @ A)
