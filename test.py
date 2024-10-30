import numpy as np
import matplotlib.pyplot as plt

# 使用对数刻度绘制
plt.plot(x, y1, 'g-', label='Line 1 (0 to 0.3)')
plt.plot(x, y2, 'b-', label='Line 2 (0 to 0.005)')
plt.yscale('log')  # y轴使用对数刻度

# 添加标题和图例
plt.xlabel('x')
plt.ylabel('Log Scale for y')
plt.title("Logarithmic Scale Comparison")
plt.legend()
plt.show()
