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

    def error(self, predictions, labels):
        error_rate = 1
        errors = 0
        if (len(predictions) != len(labels)):
            print 'Different number of labels and predictions...'
        else:
            for x in range(len(predictions)):
                if predictions[x] != labels[x]:
                    errors += 1
                    
            error_rate = float(errors) / len(predictions)
        return error_rate

         
class PerceptronBias(Perceptron) :

    """An implementation of the perceptron algorithm with bias."""

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

        scores = np.dot(w, X) + bias            # IS THIS THE CORRECT THING TO DO WITH BIAS???
        return np.sign(scores)


class PerceptronPocket(PerceptronBias) :

    """An implementation of the perceptron algorithm w/ bias which tracks 
    the weight vector which is 'best-so-far'. """

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


        self.w = np.zeros(len(X[0]))
        self.pocket = self.w            # initialize the pcoket weight vector
        self.pocket_error = 1
        converged = False
        iterations = 0
        while (not converged and iterations < self.max_iterations) :
            converged = True
            for i in range(len(X)) :
                if y[i] * self.discriminant(X[i]) <= 0 :                
                    self.w = self.w + y[i] * self.learning_rate * X[i]
                    converged = False

                    predictions = [self.predict(i[1:], False) for i in X]          # strip bias terms from features, then predict using self.w
                    error_now = self.error(predictions, y)
                    if error_now < self.pocket_error:
                        self.pocket = np.array(self.w)
                        self.pocket_error = error_now


            iterations += 1
        self.converged = converged
        if converged :
            print 'converged in %d iterations ' % iterations
            

    def predict(self, X, use_pocket=True) :
        """
        make predictions using a trained linear classifier

        Parameters
        ----------

        X : ndarray, shape (num_examples, n_features)
        Training data.

        use_pocket : Boolean, set to True if using self.pocket, otherwise uses self.w
        """

        if (use_pocket == True):
            bias = self.pocket[0]
            w = self.w[1:]
        else:
            bias = self.w[0]
            w = self.w[1:]

        scores = np.dot(w, X) + bias            # IS THIS THE CORRECT THING TO DO WITH BIAS???
        return np.sign(scores)



class PerceptronModified(Perceptron) :

    """An implementation of the perceptron algorithm. This modified version
    updats the weight vector based on the data point that maximizes the given function lambda.
    Note that this implementation does not include a bias term."""

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
        maximum_initial_w_value = 1
        c = 0                           # init c and makes sure (0 < c < 1), not 0
        while (c == 0):
            c = np.random.uniform()

        self.w = np.random.uniform(size=len(X[0])) * maximum_initial_w_value

        converged = False
        iterations = 0
        while (not converged and iterations < self.max_iterations) :
            converged = True

            
            #evaluate lambda and put in tuple (i, lambda_value) in list 'lambdas'
            lambdas = [(i, y[i] * np.dot(self.w, X[i])) for i in range(len(X))]


            
            #filter for those i where lambda < c||w||
            all_eligible = []
            for lam in lambdas:
                if (lam[1] < c * self.norm(self.w)):
                    all_eligible.append(lam)

            # print all_eligible, len(all_eligible)


            # choose j (from all eligible i) for which lambda is maximized
            j = ''

            all_lambdas = [lam[1] for lam in all_eligible]
            
            if (all_lambdas == []):
                # none are eligible
                converged = True

            else:
                max_lambda = max(all_lambdas)

                for lam in all_eligible:
                    if (lam[1] == max_lambda):
                        j = lam[0]
                        break

            if (j != ''):
                #update w
                self.w = self.w + y[j] * self.learning_rate * X[j]
                converged = False
            else:
                converged = True

            # stop if no more corrections are made
            # if (False):
            #     converged = True

            iterations += 1

        self.converged = converged
        if converged :
            print 'converged in %d iterations ' % iterations

    def norm(self, x):
        return np.sqrt(np.sum(np.square(x)))



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