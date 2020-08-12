import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetworkClassifier:
    '''Implementation of 2-layer neural network for binary classification'''
    
    def __init__(self, n_neurons=4, n_iterations=10000, learning_rate=0.01, verbose=False, random_state=17):
        self.n_neurons = n_neurons              # number of neurons in hidden layer
        self.n_iterations = n_iterations        # number of iterations in gradient descent
        self.learning_rate = learning_rate      # learning rate (gradient step)
        self.verbose = verbose                  # print cost on some steps
        self.random_state = random_state        # random seed
        self.W1 = None                          # weights for hidden layer
        self.b1 = None                          # bias for hidden layer
        self.W2 = None                          # weights for output layer
        self.b2 = None                          # bias for output layer
        self.costs = None                       # cost per every 500th iteration


    def fit(self, X, y):
        ''' X - np.array of training examples of size (n_examples, n_features) 
            y - np.array of labels (0 or 1) of size (1, n_examples)'''
        
        X_train = X.T
        y_train = y
        
        n_examples = X_train.shape[1]
        n_x = X_train.shape[0]  # size of input layer
        n_h = self.n_neurons    # size of hidden layer
        n_y = y_train.shape[0]  # size of output layer
        
        # initialize network parameters
        np.random.seed(self.random_state)
        self.W1 = np.random.randn(n_h, n_x) * 0.01
        self.b1 = np.zeros((n_h, 1))
        self.W2 = np.random.randn(n_y, n_h) * 0.01
        self.b2 = np.zeros((n_y, 1))
        self.costs = list()
            
        # run gradient descent trying minimize cost
        for i in np.arange(self.n_iterations):
            # forward propagation
            Z1 = self.W1 @ X_train + self.b1   # linear part of hidden layer
            A1 = np.tanh(Z1)                   # result of hidden layer after activation function
            Z2 = self.W2 @ A1 + self.b2        # linear part of output layer
            A2 = sigmoid(Z2)                   # result of ouput layer after sigmoid function
            
            # calculate current cost
            cost = -np.mean(y_train * np.log(A2) + (1 - y_train) * np.log(1 - A2))
            
            # backward propagation (calculate gradients with respect to different parameters)
            dZ2 = A2 - y_train
            dW2 = (dZ2 @ A1.T) / n_examples
            db2 = np.sum(dZ2, axis=1, keepdims=True) / n_examples
            dZ1 = (self.W2.T @ dZ2) * (1 - np.power(A1, 2))
            dW1 = (dZ1 @ X_train.T) / n_examples
            db1 = np.sum(dZ1, axis=1, keepdims=True) / n_examples
            
            # update parameters
            self.W1 = self.W1 - self.learning_rate * dW1
            self.b1 = self.b1 - self.learning_rate * db1
            self.W2 = self.W2 - self.learning_rate * dW2
            self.b2 = self.b2 - self.learning_rate * db2
            
            # Record the costs
            if i % 500 == 0:
                self.costs.append(cost)
        
            # Print the cost every 500 training iterations
            if self.verbose and i % 500 == 0:
                print ("Cost after iteration %i: %f" % (i, cost))


    def predict_proba(self, X):
        '''X - np.array of training examples of size (n_examples, n_features)
        
           Return array with predicted probabilities of size (1, n_samples)'''
        
        Z1 = self.W1 @ X.T + self.b1
        A1 = np.tanh(Z1)
        Z2 = self.W2 @ A1 + self.b2
        probabilities = sigmoid(Z2)
        return probabilities
    
    
    def predict(self, X, treshhold=0.5):
        '''X - np.array of training examples of size (n_examples, n_features)
        
           Return array with predicted labels (0 or 1) of size (1, n_samples)'''
        
        probabilities = self.predict_proba(X)
        prediction = (probabilities > treshhold).astype(float)
        return prediction
