import numpy as np #linear algebra library
"""
Essentially trying to compute these outputs from these inputs

  Inputs	  Output
0	0	1		0
1	1	1		1
1	0	1		1
0	1	1		0
"""
#This is the sigmoid function, maps any value to a value between 0 and 1
def sigmoid(x, derivative = False):
	if(derivative == True):
		return x * (1-x)

	else:
		return 1 / (1 + np.exp(-x))

def tanh(x, derivative = False):
	if(derivative == True):
		x = np.tan(x)
		return 1 - (x * x)

	else:
		return np.tanh(x)

#input matrix (3 input nodes and 4 training examples)
x = np.array([[0, 0, 1],
			  [0, 1, 1],
			  [1, 0, 1], 
			  [1, 1, 1]])

#output matrix (Transposed to make it 4 rows and 1 column)
#Since each column is an output node and we have 1 column, the betwork is 3 inputs and 1 output
y = np.array([[0, 0, 1, 1]]).T

np.random.seed(5) #good practice to seed random numbers so they are always generated the same way

#At first, weights are initialized randomly with mean 0
#Interesting to note that the neural network is really just this matrix of weights
#This is a 3 x 1 matrix of weights
weights1 = 2 * np.random.random((3, 1)) - 1

for interation in xrange(10000):
	#Forward propagation

	#First layer is simply data, so its explicitly described at this point
	inputLayer = x

	#Let network try to "predict" the output given the input to see how it performs
	#We take the dot product of an instance of the input with the weights and put it through the sigmoid function and so with 4 inputs, we get 4 guesses
	guess = tanh(np.dot(inputLayer, weights1))

	#Checking how errornous the guess was by subtracting the actual answer from it
	error = guess - y

	# multiply how much we missed by the
	# slope of the sigmoid at the values in guess
	# a delta is a variation of a function or variable
	guess_delta = error * tanh(guess, True)

	weights1 += np.dot(inputLayer.T, guess_delta)

print "Output after training: "
for i in guess:
	print '{:.20f}'.format(i[0])
