import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    '''Implementation of logistic regression'''
    
    def __init__(self, n_iterations=10000, learning_rate=0.01, verbose=False, random_state=None):
        self.n_iterations = n_iterations        # number of iterations in gradient descent
        self.learning_rate = learning_rate      # learning rate (gradient step)
        self.verbose = verbose                  # print cost on some steps
        self.random_state = random_state        # random seed
        self.W = None                           # weights for output layer
        self.b = None                           # bias for output layer
        self.costs = None                       # cost per iteration
        
    
    def fit(self, X, y):
        ''' X - np.array of training examples of size (n_examples, n_features) 
            y - np.array of labels (0 or 1) of size (1, n_examples)'''

        X_train = X.T
        y_train = y

        n_examples = X_train.shape[1]
        n_features = X_train.shape[0]
        
        # initialize parameters
        if self.random_state is None:
            self.W = np.zeros((n_features, 1))
        else:
            np.random.seed(self.random_state)
            self.W = np.random.randn(n_features, 1) * 0.01
        self.b = 0.
        self.costs = list()
        
        # run gradient descent trying minimize cost
        for i in np.arange(self.n_iterations):
            # forward propagation
            pred = sigmoid(self.W @ X_train + self.b)
            
            # calculate current cost
            cost = -np.mean(y_train * np.log(pred) + (1 - y_train) * np.log(1 - pred))
            
            # backward propagation
            dW = (X_train @ (pred - y_train).T) / n_examples
            db = np.mean(pred - y_train)
            
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
            
            # Record the costs
            if i % 500 == 0:
                self.costs.append(cost)
        
            # Print the cost every 500 training iterations
            if self.verbose and i % 500 == 0:
                print ("Cost after iteration %i: %f" % (i, cost))


    def predict_proba(self, X):
        '''X - np.array of training examples of size (n_examples, n_features)
           
           Return array with predicted probabilities of size (1, n_samples)'''

        probabilities = sigmoid(self.W @ X.T + self.b)
        return probabilities
    
    
    def predict(self, X, treshhold=0.5):
        '''X - np.array of training examples of size (n_examples, n_features)
        
           Return array with predicted labels (0 or 1) of size (1, n_samples)'''
        
        probabilities = self.predict_proba(X)
        prediction = (probabilities > treshhold).astype(float)
        return prediction