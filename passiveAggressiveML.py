import numpy as np
import matplotlib.pyplot as plt

def getData(n):
	mean1 = [0, 0]
	cov1 = [[1, 0], [0, 1]]

	mean2 = [1.5, 1.5]
	cov2 = [[1, 0], [0, 1]]

	x1, y1 = np.random.multivariate_normal(mean1, cov1, n).T
	x2, y2 = np.random.multivariate_normal(mean2, cov2, n).T

	x = np.ones((2*n, 3))
	x[0:n, 0] = x1
	x[0:n, 1] = y1
	x[n:2*n, 0] = x2
	x[n:2*n, 1] = y2

	y = np.ones((2*n, 1))
	y[0:n] = -1

	return x, y

n = 50

x, y = getData(n)

# Initial w
w = [0, 0, 0]

# Number of iterations
T = 100

# Random permutation of data
perm = np.random.permutation(len(y))
reordered_x = x[perm]
reordered_y = y[perm]

for t in range(T):
	x_t = reordered_x[t, :]
	yhat_t = np.dot(w, x_t)
	y_t = reordered_y[t]
	l_t = max(0, 1-y_t*yhat_t)
	T_t = l_t/(np.sum(x_t**2))
	w = w + T_t*y_t*x_t

	# Classify all points and display results
	yhat = np.zeros(y.shape)
	numCorrect = 0
	for j in range(len(y)):
		x_j = x[j, :]
		y_j = y[j]
		yhat[j] = np.dot(w, x_j)
		if(y[j]*yhat[j] > 0):
			numCorrect = numCorrect + 1
	print("Epoch {} Accuracy {}".format(t, numCorrect/len(y)))
	colorvec1 = []
	for j in range(n):
		if(y[j]*yhat[j] > 0):
			colorvec1.append('b')
		else:
			colorvec1.append('r')
	colorvec2 = []
	for j in range(n):
		if(y[j+n]*yhat[j+n] > 0):
			colorvec2.append('b')
		else:
			colorvec2.append('r')

	# Make this point green
	if(perm[t]<n):
		colorvec1[perm[t]] = 'g'
	else:
		colorvec2[perm[t]-n] = 'g'

	# Calculate seperating line
	plotx = np.array([-3, 4])
	ploty = (-w[2] - w[0]*plotx)/w[1]

	plt.figure(dpi=300)
	plt.scatter(x[0:n,0], x[0:n,1], s=50, marker='s', color=colorvec1)
	plt.scatter(x[n:2*n, 0], x[n:2*n, 1], s=50, marker='X', color=colorvec2)
	plt.plot(plotx, ploty)
	plt.axis([-2.9, 3.9, -2.9, 3.9])
	plt.title('Iteration {}'.format(t))
	plt.savefig('iterationOutputs/iter{}.png'.format(str(t).zfill(3)))
	plt.close()
