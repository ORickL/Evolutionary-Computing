from numpy import exp, ndarray
import sys

sys.path.insert(0, 'evoman') 
from controller import Controller

def sigmoid_activation(x: ndarray) -> ndarray:
	return 1. / (1. + exp(-x))


def normalize_inputs(inputs: ndarray) -> ndarray:
	in_min = min(inputs)
	in_max = max(inputs)
	abs_max = max(abs(in_min), abs(in_max))
	scaling_factor = 1. / abs_max
	return scaling_factor * inputs


"""
implements controller structure for player.
It is heavily inspired by the demo controller but with a small
change in the normalizing inputs 
"""
class my_controller(Controller):
	def __init__(self, _n_hidden: int):
		# Number of hidden neurons
		self.n_hidden = _n_hidden

	def control(self, inputs: ndarray, controller: ndarray):
		"""
		The player has 5 possible actions, so the output layer will be of length 5.
		len(inputs) determine the number of input layers.
		We will want 1 hidden layer (of self.n_hidden size) between the input and output. This hidden layer will
		be the encoding of the genome. Seems easy enough.

		This segment is taken from the demo_controller.py implementation.
		"""
		inputs = normalize_inputs(inputs)
		# [left, right, jump, shoot, release]
		if self.n_hidden > 0:
			# Preparing the weights and biases from the controller of layer 1

			# Biases for the n hidden neurons
			bias1 = controller[:self.n_hidden].reshape(1,self.n_hidden)
			# Weights for the connections from the inputs to the hidden nodes
			weights1_slice = len(inputs)*self.n_hidden + self.n_hidden
			weights1 = controller[self.n_hidden:weights1_slice].reshape((len(inputs),self.n_hidden))

			# Outputs activation first layer.
			output1 = sigmoid_activation(inputs.dot(weights1) + bias1)

			# Preparing the weights and biases from the controller of layer 2
			bias2 = controller[weights1_slice:weights1_slice + 5].reshape(1,5)
			weights2 = controller[weights1_slice + 5:].reshape((self.n_hidden,5))

			# Outputting activated second layer. Each entry in the output is an action
			output = sigmoid_activation(output1.dot(weights2)+ bias2)[0]
			# print(f"output1: {output1}, output: {output}")
		else:
			bias = controller[:5].reshape(1, 5)
			weights = controller[5:].reshape((len(inputs), 5))

			output = sigmoid_activation(inputs.dot(weights) + bias)[0]

		# takes decisions about sprite actions
		if output[0] > 0.5:
			left = 1
		else:
			left = 0

		if output[1] > 0.5:
			right = 1
		else:
			right = 0

		if output[2] > 0.5:
			jump = 1
		else:
			jump = 0

		if output[3] > 0.5:
			shoot = 1
		else:
			shoot = 0

		if output[4] > 0.5:
			release = 1
		else:
			release = 0

		return [left, right, jump, shoot, release]

	def debug(self):
		print("Debugging", self.n_hidden)