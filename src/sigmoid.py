import numpy as np
import matplotlib.pyplot as plt

def sigmoid(a):
	res = []
	for e in a:
		res.append(1/(1+np.exp(-e)))
	return res

a = list(range(-20,20,1))
print(sigmoid(a))

plt.plot(sigmoid(a))
plt.show()