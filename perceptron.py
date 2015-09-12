import numpy as np
from matplotlib import pyplot as plt

class Perceptron :

    """An implementation of the perceptron algorithm.
    Note that this implementation does not include a bias term"""

    def __init__(self, max_iterations=100, learning_rate=0.2) :

        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    def fit(self, X, y) :
        """
        Train a classifier using the perceptron training algorithm.
        After training the attribute 'w' will contain the perceptron weight vector.

        Parameters
        ----------

        X : ndarray, shape (num_examples, n_features)
        Training data.

        y : ndarray, shape (n_examples,)
        Array of labels.
        
        """
        self.w = np.zeros(len(X[0]))
        converged = False
        iterations = 0
        while (not converged and iterations < self.max_iterations) :
            converged = True
            for i in range(len(X)) :
                if y[i] * self.discriminant(X[i]) <= 0 :
                    #print 'y[i]', y[i]
                    #print 'learning_rate', self.learning_rate
                    #print 'w', self.w, len(self.w), np.shape(self.w)
                    #print 'X[i]', X[i], len(X[i]), np.shape(X[1])
                    
                    self.w = self.w + y[i] * self.learning_rate * X[i]
                    converged = False
                    #plot_data(X, y, self.w)
            iterations += 1
        self.converged = converged
        if converged :
            print 'converged in %d iterations ' % iterations

    def discriminant(self, x) :
        return np.dot(self.w, x)
            
    def predict(self, X) :
        """
        make predictions using a trained linear classifier

        Parameters
        ----------

        X : ndarray, shape (num_examples, n_features)
        Training data.
        """
        
        scores = np.dot(self.w, X)
        return np.sign(scores)
        
class PerceptronBias :

    """An implementation of the perceptron algorithm with bias."""

    def __init__(self, max_iterations=100, learning_rate=0.2) :

        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    def fit(self, Xinput, y) :
        """
        Train a classifier using the perceptron training algorithm with bias.
        After training the attribute 'w' will contain the perceptron weight vector of length equal to len(X) + 1.

        Parameters
        ----------

        X : ndarray, shape (num_examples, n_features)
        Training data.

        y : ndarray, shape (n_examples,)
        Array of labels.
        
        """

        X = []

        # Hide bias here in extra term, set to 1
        for i in range(len(Xinput)):
            X.append(np.insert(Xinput[i],0,1))
            #print X[i]

        self.w = np.zeros(len(X[0]))
        converged = False
        iterations = 0
        while (not converged and iterations < self.max_iterations) :
            converged = True
            for i in range(len(X)) :
                if y[i] * self.discriminant(X[i]) <= 0 :                
                    self.w = self.w + y[i] * self.learning_rate * X[i]
                    converged = False
                    #plot_data(X, y, self.w)
            iterations += 1
        self.converged = converged
        if converged :
            print 'converged in %d iterations ' % iterations

    def discriminant(self, x) :
        return np.dot(self.w, x)
            
    def predict(self, X) :
        """
        make predictions using a trained linear classifier

        Parameters
        ----------

        X : ndarray, shape (num_examples, n_features)
        Training data.
        """

        bias = self.w[0]
        w = self.w[1:]
        
        # DO SOMETHING WITH BIAS HERE

        scores = np.dot(w, X) + bias
        return np.sign(scores)


class PerceptronPocket :

    """An implementation of the perceptron algorithm w/ bias which tracks 
    the weight vector which is 'best-so-far'. """

    def __init__(self, max_iterations=100, learning_rate=0.2) :

        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    def fit(self, X, y) :
        """
        Train a classifier using the perceptron training algorithm with bias.
        After training the attribute 'w' will contain the perceptron weight vector of length equal to len(X) + 1.

        Parameters
        ----------

        X : ndarray, shape (num_examples, n_features)
        Training data.

        y : ndarray, shape (n_examples,)
        Array of labels.
        
        """
        # Hide bias here in extra term, set to 1
        for i in range(len(X)):
            np.insert(X[i],0,1)


        self.w = np.zeros(len(X[0]))
        self.pocket = self.w            # initialize the pcoket weight vector
        converged = False
        iterations = 0
        while (not converged and iterations < self.max_iterations) :
            converged = True
            for i in range(len(X)) :
                if y[i] * self.discriminant(X[i]) <= 0 :                
                    self.w = self.w + y[i] * self.learning_rate * X[i]
                    converged = False
                    #plot_data(X, y, self.w)

                    if new_error() < old_error():
                        self.pocket = self.w 


                # test pocket vs new w



            iterations += 1
        self.converged = converged
        if converged :
            print 'converged in %d iterations ' % iterations

    def discriminant(self, x) :
        return np.dot(self.w, x)
            
    def predict(self, X) :
        """
        make predictions using a trained linear classifier

        Parameters
        ----------

        X : ndarray, shape (num_examples, n_features)
        Training data.
        """
        
        scores = np.dot(self.w, X)
        return np.sign(scores)

def generate_separable_data(N) :
    w = np.random.uniform(-1, 1, 2)
    print w,w.shape
    X = np.random.uniform(-1, 1, [N, 2])
    print X,X.shape
    y = np.sign(np.dot(X, w))
    return X,y,w
    
def plot_data(X, y, w) :
    fig = plt.figure(figsize=(5,5))
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    a = -w[0]/w[1]
    pts = np.linspace(-1,1)
    plt.plot(pts, a*pts, 'k-')
    cols = {1: 'r', -1: 'b'}
    for i in range(len(X)): 
        plt.plot(X[i][0], X[i][1], cols[y[i]]+'o')
    plt.show()

if __name__=='__main__' :
    X,y,w = generate_separable_data(40)
    p = Perceptron()
    p.fit(X,y)