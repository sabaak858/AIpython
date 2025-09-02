import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-10, 10, 400)
y1 = 2 * x + 1
y2 = 2 * x + 2
y3 = 2 * x +3
plt.figure(figsize=(10, 6))
plt.plot(x, y1, 'k-', label='y = 2x + 1')
plt.plot(x, y2, 'k--', label='y = 2x + 2')
plt.plot(x, y3, 'k:', label='y = 2x + 3')
plt.title('Graphs of y = 2x + 1, y = 2x + 2, and y = 2x + 3')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()