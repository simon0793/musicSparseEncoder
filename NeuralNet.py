import numpy as np
import random
from scipy.special import expit as sigmoid
import pickle
import time

class NeuralNet:
	'''
	V: a (n_hid)-by-(n_in + 1) matrix where the (i, j)-entry represents the weight
	connecting the j-th unit in the input layer to the i-th unit in the hidden
	layer. The i-th row of V represents the ensemble of weights feeding into the
	i-th hidden unit. Note: there is an additional row for weights connecting the
	bias term to each unit in the hidden layer.

	W: a (n_out)-by-(n_hid + 1) matrix where the (i, j)-entry represents the weight
	connecting the j-th unit in the hidden layer to the i-th unit in the output
	layer. The i-th row of W represents the ensemble of weights feeding into the i-
	th output unit. Note: again there is an additional row for weights connecting
	the bias term to each unit in the output layer.

	n_in: the number of features for the input samples

	n_hid: size of the hidden layer

	n_out: number of classes
	'''
	def __init__(self, n_in = 784, n_out = 10, n_hid = 200):
		self.n_in = n_in
		self.n_out = n_out
		self.n_hid = n_hid

		self.layers = []
		self.layers.append(np.ones((n_in + 1, 1)))
		self.layers.append(np.ones((n_hid + 1, 1)))
		self.layers.append(np.ones((n_out, 1)))

		self.weights = []
		self.weights.append(np.zeros((n_hid, n_in + 1)))
		self.weights.append(np.zeros((n_out, n_hid + 1)))
		# self.V = None # weights are randomly intialized each time we train
		# self.W = None

		self.iter_count = None
		self.saved_weights_y = []
		self.saved_weights_x = []
		self.sample_weight_period = 1000

		self.most_recent_saved_weights_file = None

		self.dJ_dx_out_fn = None
		self.error_fn = None

	def set_error_function(self, error_fn):
		if error_fn == "MSE":
			self.error_fn = self.mean_squared_error
			self.dJ_dx_out_fn = lambda y, x: -(y - x)
		elif error_fn == "CEE":
			self.error_fn = self.cross_entropy_error
			self.dJ_dx_out_fn = lambda y, x: -(np.divide(y,x) - np.divide(1.0-y,1.0-x))
		else:
			print("ERROR: Invalid error fn")
			self.error_fn = None

	@staticmethod
	def fast_sigmoid_grad(x):
		'''
		Result from using Wolfram Alpha
		>>> '%.6f' % NeuralNet.fast_sigmoid_grad(0.5)
		'0.235004'
		>>> NeuralNet.fast_sigmoid_grad(500)
		7.1245764067412855e-218

		# >>> NeuralNet.fast_sigmoid_grad(-500) # should be 7.1245764067412855e-218 but will fail since not stable near tail
		# 0.0
		'''
		ex = np.exp(-x)
		return ex/((1 + ex)**2)

	'''
	Better tail performance, see http://stackoverflow.com/questions/21106134/numpy-pure-functions-for-performance-caching
	'''
	@staticmethod
	def safe_sigmoid_grad(x):
		'''
		Result from using Wolfram Alpha
		>>> '%.6f' % NeuralNet.safe_sigmoid_grad(0.5)
		'0.235004'
		>>> NeuralNet.safe_sigmoid_grad(500)
		7.1245764067412855e-218
		>>> NeuralNet.safe_sigmoid_grad(-500) # should be 7.1245764067412855e-218 but will fail since not stable near tail
		7.1245764067412855e-218
		'''
		return (0.5 / np.cosh(0.5*x))**2

	@staticmethod
	def tanh_grad(x):
		'''
		'''
		return 1.0 - np.square(np.tanh(x))

	'''
	y: array of ground truth classes with length n_out and a 1 in the true class and a 0 everywhere else
		i.e [[1.0 0.0 0.0],...,[0.0 1.0 0.0]]
	z_x: array of computed outputs with length n_out
	'''
	@staticmethod
	def mean_squared_error(y, z_x):
		'''
		# >>> y = np.array([1.0, 0.0, 0.0])
		# >>> z_x = np.array([0.5, 0.5, 0.0])
		# >>> str(NeuralNet.mean_squared_error(y, z_x))
		# '0.25'
		# >>> y = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
		# >>> z_x = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
		# >>> str(NeuralNet.mean_squared_error(y, z_x))
		# '[0.25, 0.25]'
		'''
		return np.dot(0.5, np.sum(np.square(y - z_x), 1))

	@staticmethod
	def cross_entropy_error(y, z_x):
		'''
		>>> y = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
		>>> z_x = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
		>>> str(NeuralNet.cross_entropy_error(y, z_x))
		'[0.25, 0.25]'
		'''
		z_x = np.clip(z_x, 1e-32, 0.9999999999999999)
		return -(np.sum(np.multiply(y, np.log(z_x)) + np.multiply((1.0 - y), np.log(1.0 - z_x)), 1))

	'''
	design_matrix: each row is a sample and the columns are the features
	labels: each row is the label for the corresponding sample, in the form [0, 0, 0, ..., 1, 0, ..., 0]
	'''
	def train(self, design_matrix, labels, mean, std_dev, epsilon, num_epochs):
		'''
		# >>> import numpy as np
		# >>> nn = NeuralNet(n_in = 2, n_out = 1, n_hid = 2)
		# >>> nn.set_error_function("MSE")
		# >>> dm = np.array([[0,0], [0,1], [1,0], [1,1]])
		# >>> labels = np.array([[0], [1], [1], [0]])
		# >>> nn.train(dm, labels)
		# >>> print(nn.weights)
		'''
		if self.error_fn is None:
			print("ERROR: No error function selected")
			return None
		'''
		1. Initialize all weights, V;W at random
		2. while (some stopping criteria):
		3. pick one data point (x; y) at random from the training set
		4. perform forward pass (computing necessary values for gradient descent update)
		5. perform backward pass (again computing necessary values)
		6. perform stochastic gradient descent update
		7. return V;W
		'''
		self.iter_count = 0
		self.weights[0] = std_dev * np.random.randn(*self.weights[0].shape) + mean
		self.weights[1] = std_dev * np.random.randn(*self.weights[1].shape) + mean
		self.saved_weights_y = []
		self.saved_weights_x = []
		self.sample_weight_period = (design_matrix.shape[0])/50

		for i in range(num_epochs):
			if i == 1:
				self.sample_weight_period = (design_matrix.shape[0]*(num_epochs - 1))/50
			self.train_one_epoch(design_matrix, labels, epsilon)

	def continue_training(self, design_matrix, labels, epsilon, num_epochs):
		if self.error_fn is None:
			print("ERROR: No error function selected")
			return None
		'''
		1. Initialize all weights, V;W at random
		2. while (some stopping criteria):
		3. pick one data point (x; y) at random from the training set
		4. perform forward pass (computing necessary values for gradient descent update)
		5. perform backward pass (again computing necessary values)
		6. perform stochastic gradient descent update
		7. return V;W
		'''
		self.sample_weight_period = (design_matrix.shape[0]*num_epochs)/3
		for i in range(num_epochs):
			self.train_one_epoch(design_matrix, labels, epsilon)

	def train_one_epoch(self, design_matrix, labels, epsilon):
		training_indices = list(range(design_matrix.shape[0]))
		random.shuffle(training_indices)
		for i in training_indices:
			self.iter_count += 1
			self.layers[0][:-1] = design_matrix[i, :, np.newaxis]
			self.forwardPass()
			gradient_of_loss = self.backwardPass(labels[i, :, np.newaxis])
			self.stochasticGradientDescentUpdate(epsilon, gradient_of_loss)
			if self.iter_count % self.sample_weight_period == 0:
				self.saved_weights_x.append(self.iter_count)
				self.saved_weights_y.append([np.copy(self.weights[0]), np.copy(self.weights[1])])

	def predict(self, design_matrix):
		'''
		1. Compute labels of all images using the weights, V;W
		2. return labels
		'''
		training_indices = list(range(design_matrix.shape[0]))
		result = np.zeros((design_matrix.shape[0], self.layers[2].shape[0]))
		for i in training_indices:
			self.layers[0][:-1] = design_matrix[i, :, np.newaxis]
			self.forwardPass()
			result[i, :] = np.transpose(self.layers[2])
		return result

	def save_snapshot(self):
		self.most_recent_saved_weights_file = time.strf("%d-%H-%m") + "NNweights.p"
		pickle.dump(self.saved_weights, open(self.most_recent_saved_weights_file, "wb"))

	def load_snapshot(self, weight_file=None):
		if weight_file is None:
			weight_file = self.most_recent_saved_weights_file
		self.weights = pickle.load(open(weight_file, "rb"))

	'''
	Compute all the node values and store them for use in the backwardPass and stochasticGradientDescentUpdate
	'''
	def forwardPass(self):
		'''
		>>> import numpy as np
		>>> nn = NeuralNet(n_in = 2, n_out = 1, n_hid = 2)
		>>> nn.weights[0] = np.array([[1.0, 1.0, 0.5], [-1.0, -1.0, -1.5]])
		>>> nn.weights[1] = np.array([[1.0, 1.0, 1.5]])
		>>> nn.layers[0][0] = 1.0
		>>> nn.layers[0][1] = 0.0
		>>> nn.forwardPass()
		>>> assert(nn.layers[2][0,0] > 0.5)
		>>> nn.layers[0][0] = 0.0
		>>> nn.layers[0][1] = 1.0
		>>> nn.forwardPass()
		>>> assert(nn.layers[2][0,0] > 0.5)
		>>> nn.layers[0][0] = 0.0
		>>> nn.layers[0][1] = 0.0
		>>> nn.forwardPass()
		>>> assert(nn.layers[2][0,0] > 0.5)
		>>> nn.layers[0][0] = 1.0
		>>> nn.layers[0][1] = 1.0
		>>> nn.forwardPass()
		>>> assert(nn.layers[2][0,0] > 0.5)
		'''
		self.layers[1][:-1] = np.tanh(np.dot(self.weights[0], self.layers[0]))
		self.layers[2] = sigmoid(np.dot(self.weights[1], self.layers[1]))
		return None

	'''
	Compute the gradient_of_loss for each layer of weights, starting at last layer of weights and working backwards
	y: dimension is same as the output layer
	'''
	def backwardPass(self, y):
		'''
		>>> import numpy as np
		>>> nn = NeuralNet(n_in = 2, n_out = 1, n_hid = 2)
		>>> nn.set_error_function("MSE")
		>>> nn.weights[0] = np.array([[1.0, 1.0, 0.5], [-1.0, -1.0, -1.5]])
		>>> nn.weights[1] = np.array([[1.0, 1.0, 1.5]])
		>>> nn.layers[0][0] = 1.0
		>>> nn.layers[0][1] = 0.0
		>>> nn.forwardPass()
		>>> y = np.array([0.0])
		>>> res = nn.backwardPass(y)
		'''
		gradient_of_loss = [None, None]
		delta = [None, None]
		# Hidden to Out
		self.layers[2] = np.clip(self.layers[2], 1e-32, 0.9999999999999999)
		dJ_dx_out = self.dJ_dx_out_fn(y, self.layers[2])
		delta[1] = np.multiply(dJ_dx_out, NeuralNet.fast_sigmoid_grad(np.dot(self.weights[1], self.layers[1])))
		gradient_of_loss[1] = np.dot(delta[1], np.transpose(self.layers[1]))
		# In to Hidden
		delta[0] = np.multiply(np.dot(np.transpose(self.weights[1][:, :-1]), delta[1]), NeuralNet.tanh_grad(np.dot(self.weights[0], self.layers[0])))
		gradient_of_loss[0] = np.dot(delta[0], np.transpose(self.layers[0]))
		return gradient_of_loss

	def stochasticGradientDescentUpdate(self, epsilon, gradient_of_loss):
		# w <- w - epsilon*gradient_of_loss
		# print(gradient_of_loss)
		self.weights[0] -= np.dot(epsilon, gradient_of_loss[0])
		self.weights[1] -= np.dot(epsilon, gradient_of_loss[1])

if __name__ == "__main__":
    import doctest
    doctest.testmod()