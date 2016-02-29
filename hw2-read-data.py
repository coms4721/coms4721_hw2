from scipy.io import loadmat
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

data = loadmat('spam_fixed.mat')

Ytrain = data['labels'].flatten()
Xtrain = data['data']
Ytest  = data['testlabels'].flatten()
Xtest  = data['testdata']


class Perceptron(object):
	def fit(self, X, Y):
		N, D = X.shape
		V = [np.zeros(D)]
		C = [0]
		for t in xrange(64):
			for i in xrange(N):
				v = V[-1]
				y_hat_i = np.sign(v.dot(X[i]))
				if y_hat_i == Y[i]:
					C[-1] = C[-1] + 1
				else:
					new_v = v + Y[i]*X[i]
					V.append(new_v)
					C.append(1)
				# y_hat_i = sign(v dot x_i)
				# if y_hat_i ==? y[i]...
		self.v = np.zeros(D)
		total_votes = 0
		for c, v in zip(C[1:], V[1:]):
			self.v += c*v
			total_votes += c
		self.v /= total_votes


	def score(self, X, Y):
		P = np.sign(X.dot(self.v))
		return np.mean(P == Y)


class BigPerceptron(object):
	def fit(self, X, Y):
		X2 = transform(X)
		self.model = Perceptron()
		self.model.fit(X2, Y)

	def score(self, X, Y):
		X2 = transform(X)
		return self.model.score(X2, Y)

def transform(X):
	N, D = X.shape
	X2 = np.zeros((N, 1710))
	X2[:,:D] = X
	j = D
	for i in xrange(D):
		for k in xrange(D):
			if i <= k:
				X2[:,j] = X[:,i]*X[:,k]
				j += 1
	# mu = X2.mean(axis=0)
	# std = X2.std(axis=0)
	# X2 = (X2 - mu) / std
	return X2


class BigLogistic(object):
	def fit(self, X, Y):
		X2 = transform(X)
		self.model = LogisticRegression()
		self.model.fit(X2, Y)

	def score(self, X, Y):
		X2 = transform(X)
		return self.model.score(X2, Y)


def cross_validation(model, X, Y):
	# split the data into 10 parts
	N = len(Y)
	batchsize = N / 10 + 1
	scores = []
	for i in xrange(10):
		# test on i-th part, train on other 9 parts
		# (i + 1)*batchsize
		start = i*batchsize
		end = (i*batchsize + batchsize)
		Xvalid = X[start:end]
		Yvalid = Y[start:end]

		Xtrain = np.concatenate([  X[:start] , X[end:] ])
		Ytrain = np.concatenate([  Y[:start] , Y[end:] ])

		model.fit(Xtrain, Ytrain)
		scores.append(model.score(Xvalid, Yvalid))
	return np.mean(scores)


models = {
	'logistic': LogisticRegression(),
	'randomforest': RandomForestClassifier(),
	'perceptron': Perceptron(),
	'biglogistic': BigLogistic(),
	'bigperceptron': BigPerceptron(),
}

def main():
	for name, model in models.iteritems():
		print "Model:", name, "accuracy:", cross_validation(model, Xtrain, Ytrain)

if __name__ == '__main__':
	main()
