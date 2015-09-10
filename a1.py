# a1.py - 
# by K. Brett Mulligan
# CSU - CS545 (Fall 2015)
# Assignment 1

# Note: Write custom sign function


from __future__ import division

import math
import numpy as np
from matplotlib import pyplot as plt


import perceptron as per

max_iterations = 2000
learning_rate = 0.01
reserve = 0.1      # percent of examples to reserve for testing

heart_file = "heart.csv"
gisette_file = "gisette_train.data"


def normalize(a):
    """Returns normalized 1-dimensional vector z from vector a. 
       Finds largest magnitude in a and divides each member of a that value."""
    # all abs
    b = [abs(x) for x in a]
    largest = max(b)
    z = [x/largest for x in a]
    return z

def extract(examples):
    ids = [patient[0] for patient in examples]
    labels = [patient[1] for patient in examples]
    raw_data = [patient[2:] for patient in examples]
    
    return labels, raw_data
    
def accuracy(scores, labels):
    acc = 0
    correct = 0
    if (len(scores) != len(labels)):
        print 'Different number of labels and scores...'
    else:
        for x in range(len(scores)):
            if scores[x] == labels[x]:
                correct += 1
                
        acc = correct / len(scores)
    return acc

if __name__ == '__main__':
    print 'Testing...a1.py'
    
    #a = np.arange(10) - 7
    #print a
    #print normalize(a)
    
    data = np.genfromtxt(heart_file, delimiter=",", comments="#")
    #X = np.genfromtxt(gisette_file)

    #print data
    #print len(data)
    #print len(data[0])

    num_for_testing = math.floor(len(data) * reserve)
    
    #np.random.shuffle(data)
    training = data[num_for_testing:]
    testing = data[:num_for_testing]
    
    print len(data), 'examples loaded.'
    print len(training), 'for training'
    print len(testing), 'for testing'
    
    labels, raw = extract(training)
    
    
    print ''
    print 'Training examples:', len(raw)
    print 'Features:', len(raw[0])
    
    print ''
    print 'Youngest:', min(np.transpose(raw)[0])
    print 'Oldest:', max(np.transpose(raw)[0])

    print 'Patient w/ heart disease:', len([x for x in training if x[1] > 0])
    print 'Patient w/o heart disease:', len([x for x in training if x[1] < 0])
    
    p = per.Perceptron(max_iterations, learning_rate)
    
    normalized = [normalize(x) for x in raw]
    p.fit(np.array(raw), np.array(labels))
    
    print ''
    print 'WEIGHT VECTOR'
    print p.w
    
    labels_results_train = [p.predict(i) for i in np.array(raw)]
    print ''
    print 'Training Data Accuracy:', accuracy(labels_results_train, labels)
    
    
    labels_test, raw_test = extract(testing)
    labels_results_test = [p.predict(i) for i in np.array(raw_test)]
    
    #print ''
    #print labels_test
    #print labels_results_test
    print 'Testing Data Accuracy:', accuracy(labels_results_test, labels_test)
    
    
    