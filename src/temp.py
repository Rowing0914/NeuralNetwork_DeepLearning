def f(a,n):
	b = 0
	for i in range(n):
		b += (1 - a)*(a**(i))
	return b

print(f(0.01, 2))