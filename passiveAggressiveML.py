import numpy as np
import matplotlib.pyplot as plt

def computeOptimalClassifierError(x, y, pt1, pt2):
	numPoints = x.shape[0]
	correctPts = 0
	for i in range(numPoints):
		if(y[i] < 0):
			if(np.linalg.norm(x[i, 0:2]-pt1) < np.linalg.norm(x[i, 0:2]-pt2)):
				correctPts = correctPts + 1
		else:
			if(np.linalg.norm(x[i, 0:2]-pt2) < np.linalg.norm(x[i, 0:2]-pt1)):
				correctPts = correctPts + 1
	return (1 - correctPts/numPoints)

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

np.random.seed(500)

x, y = getData(n)

optimalerror = computeOptimalClassifierError(x, y, [0, 0], [1.5, 1.5])*np.ones((len(y), 1))

# Initial w
w = [0, 0, 0]

# From paper, 0 = PA, 1 = PA-I, 2 = PA-II
method = 2
C = 0.01

# Number of iterations
T = 100

# Random permutation of data
perm = np.random.permutation(len(y))
x = x[perm]
y = y[perm]

errorPerIteration = []

for t in range(T):
	x_t = x[t, :]
	yhat_t = np.dot(w, x_t)
	y_t = y[t]
	l_t = max(0, 1-y_t*yhat_t)
	if(method == 1):
		T_t = min(C, l_t/(np.sum(x_t**2)))
	elif(method == 2):
		T_t = l_t/(np.sum(x_t**2) + (1/(2*C)))
	else:
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
	errorPerIteration.append(1-numCorrect/len(y))

	colorvec1 = []
	colorvec2 = []
	for j in range(2*n):
		if(y[j] < 0):
			if(j == t):
				colorvec1.append('g')
			elif(y[j]*yhat[j] > 0):
				colorvec1.append('b')
			else:
				colorvec1.append('r')
		else:
			if(j == t):
				colorvec2.append('g')
			elif(y[j]*yhat[j] > 0):
				colorvec2.append('b')
			else:
				colorvec2.append('r')

	# Calculate seperating line
	plotx = np.array([-10, 10])
	ploty = (-w[2] - w[0]*plotx)/w[1]

	plt.figure(dpi=300)
	plt.scatter(x[np.squeeze(y==-1),0], x[np.squeeze(y==-1),1], s=50, marker='s', color=colorvec1)
	plt.scatter(x[np.squeeze(y==1),0], x[np.squeeze(y==1),1], s=50, marker='X', color=colorvec2)
	plt.plot(plotx, ploty)
	plt.axis([-2.9, 3.9, -2.9, 3.9])
	#plt.axis([-2.5, 7, -2.2, 7])
	plt.title('Iteration {}'.format(t))
	plt.savefig('iterationOutputs/iter{}.png'.format(str(t).zfill(3)))
	plt.close()

plt.figure(dpi=300)
plt.plot(errorPerIteration, 'b', label='PA-II Error')
plt.plot(optimalerror, 'k', label='Optimal Classifer Error')
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.title("Test Error Over Training Time")
plt.legend()
plt.savefig('testerror.png')
plt.close()