import sys
import random
import csv
import math
import time

class Timer():
    """A simple timer, that you can start and stop"""
    def __init__(self):
        self.start = time.perf_counter()
    def start(self):
        self.start = time.perf_counter()
    def stop(self):
        self.last = self.start
        self.start = time.perf_counter()

        return self.start - self.last
    

def readcsv(fname):
    """Reads numeric data from the specified csv file.
    
    Reading a csv file with more than one column is returns
    a list of list of floats, such as [ [ 1.0, 2.0 ], [ 2.0, 2.0] ]

    Reading a csv file with one column returns a list of floats
    such as [ 1.0, 2.0 ]
    """
    import csv

    data = []
    with open(fname, 'rt') as fin:
        reader = csv.reader(fin)
        for line in reader:
            if len(line) > 1:
                data.append([float(l) for l in line])
            else:
                data.append(float(line[0]))
    return data


class Neuron():
    """A single neuron, ie, a perceptron!
    
    The perceptron takes a weighted sum of the inputs 
    (and a bias) and feeds this to an activation function.
    The activation function determines the output value of 
    the neuron.  Typical activation functions include the 
    sigmoid function and the relu.
    """
    
    def __init__(self, n_inputs, activation_fn=None):
        # The bias lives in self.weights[-1], the last index
        self.weights = [random.random() for i in range(n_inputs + 1)]  # +1 for the bias! 

    def sigmoid(self, input_values):
        """__ Part 1: Implement this __
        
        return the result of the sigmoid activation function from the input_values 
        and the weights (don't forget the bias!). This is a scalar value."""

        # summing input values * weights (finding x) for all input values
        x = 0
        for i, inval in enumerate(input_values):
            weighted_input = inval * self.weights[i]
            x += weighted_input
        # Remembering the bias!
        x += self.weights[-1]
        #  Finishing sigmoid calculation
        final_val = 1 / (1+math.e**(-x))
        return final_val

    def weight_string(self, fmt="%.3f  "):
        """Return a string representation of the weights with
        a specific format..."""
        
        f = ""
        for w in self.weights:
            f += fmt % w
      
        return "[" + f + "]"
    
    def compute_activations(self, batchX):
        """Returns a list of activations for each sample in the batch"""
        
        return [self.sigmoid(input_values) for input_values in batchX]

    def loss(self, batchX, batchY, lr=.001):
        """Compute the loss on the batch of data. """

        # from the slides: g(in) is the activation function
        # computed on the inputs
        g_in = self.compute_activations(batchX)

        # from the slides error: y - g(in)
        # y is 
        # loss is .5 * error
        return sum(.5*(y-g)**2 for (y,g) in zip(batchY, g_in))

    def accuracy(self, batchX, batchY):
        """__ Part 2: Implement this __
        
        Compute the accuracy on a batch of labeled data
        
        batchX - a list of lists; each inner list contains
                 float values describing the example data to be tested.
        batchY - a list of class labels (e.g., drawn from {0, 1}).

        return the prediction accuracy on this data
        """

        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0

        for i, test_case in enumerate(batchX):
            # Rounding activation function to 0 or 1
            neuron_decision = self.sigmoid(test_case)
            neuron_decision = 1 if neuron_decision >= .5 else 0

            if neuron_decision == 1:
                if neuron_decision == batchY[i]:
                    true_pos += 1
                else:
                    false_pos += 1
            elif neuron_decision == 0:
                if neuron_decision == batchY[i]:
                    true_neg += 1
                else:
                    false_neg += 1

        accuracy = (true_pos+true_neg) / (true_pos+true_neg+false_pos+false_neg)
        return accuracy

    def gradient(self, batchX, batchY):
        """The formula for the gradient is determined 
        by the loss function and the activation function. 
        (and, in the case of a more complex neural network,
        the network topology).
        
        Given the batch of labeled examples, compute the 
        gradient for each weight. 

        Return a list of gradients.
        """
        predictions = self.compute_activations(batchX)
        
        dw = [0.0]*len(self.weights)
        for predicted,y,x in zip(predictions, batchY, batchX):
            # for the given loss function,
            # the gradient is (predicted - y) * A' * x
            # so, changing the activation function changes A'
            grad = (predicted - y)* predicted * (1 - predicted)  
            for i,_x in enumerate(x):
                dw[i] += grad * _x
            dw[-1] += grad  # bias
        return dw
            
    def descend(self, batchX, batchY, lr=.01):
        """__ Part 3: Implement this __


        Perform gradient descent by:
        (1) calculating the gradient on the labled batch of data
        (2) update the weights by subtracting the
            learning rate * the gradient

        updates the weights of the perceptrons, but returns nothing
        """
        dw = self.gradient(batchX, batchY)
        for i, dweight in enumerate(dw):
            self.weights[i] -= dweight * lr


def stop(epochs, loss, losses):
    """__ Part 4: Implement this __

    This isn't a great stopping criterion,
    can you improve it?

    epochs - the number of epochs so far
    loss - the most recent training loss
    losses - the history of training losses for each epoch
             the most recent loss lives at losses[-1]

    returns True iff it's a good time to stop learning
    """
    if len(losses) < 2:
        return False

    if abs(losses[-1] - losses[-2]) < .01:
        return True


def train(trainX, trainY, testX, testY, batches_per=1, stoppingfn=stop):
    """Train a perceptron on the training data, and test it on the testing
    data.  Report loss and accuracy.  Gradient descent occurs after each
    batch. After each epoch, stats are reported and stopping criteria is
    checked."""

    t = Timer()
    neuron = Neuron(10)
    last = time.time()
    loss = neuron.loss(trainX, trainY)
    epochs = 0

    print("Epoch %d Loss: %f (%.2f)s"%(epochs, t.stop(), loss))
    print("   Weights: ", neuron.weights)

    losses = []
    while not stoppingfn(epochs, loss, losses):
        epochs += 1
        batchlen = len(trainX) // batches_per
        for i in range(batches_per):
            print(".", end="")
            neuron.descend(trainX[i*batchlen:(i+1)*batchlen], trainY[i*batchlen:(i+1)*batchlen])
        loss = neuron.loss(trainX, trainY)
        losses.append(loss)
        print("Epoch %d Loss: %f %.2f sec"%(epochs, loss, t.stop()))
        print("   Weights: ", neuron.weight_string())

    print("Losses:")
    for l in losses:
        print("%.3f"%l)

    print("Loss on Testing Data: %.2f"%(neuron.loss(trainX, trainY)))
    print("Accuracy on Training Data: %.2f"%(neuron.accuracy(trainX, trainY)))
    print("Accuracy on Testing Data: %.2f"%(neuron.accuracy(testX, testY)))


def main(batches):
    _X_train = readcsv('X_train.csv')
    _Y_train = readcsv('Y_train.csv')
    _X_test = readcsv('X_test.csv')
    _Y_test = readcsv('Y_test.csv')

    
    train(_X_train, _Y_train, _X_test, _Y_test, args.batches, stoppingfn=stop)
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', type=int, help='number of batches per epoch', default=1)

    args = parser.parse_args()
    main(**vars(args))
