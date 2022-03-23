import numpy as np
from datetime import datetime

N = 5000
a = np.arange(0, N * N).reshape(N, N) + 1
b = np.copy(a)

st = datetime.now()
np.dot(a, b)
et = datetime.now()
print("elapsed time :", et - st)
