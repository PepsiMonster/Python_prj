
import numpy as np


exp100 = np.random.exponential(scale=1, size=100)
np.savetxt("100exp.txt", exp100)

exp1000 = np.random.exponential(scale=1, size=1000)
np.savetxt("1000exp.txt", exp1000)

norm100 = np.random.normal(loc=0, scale=1, size=100)
np.savetxt("100norm.txt", norm100)

norm1000 = np.random.normal(loc=0, scale=1, size=1000)
np.savetxt("1000norm.txt", norm1000)

    



