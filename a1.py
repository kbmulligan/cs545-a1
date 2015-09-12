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

max_iterations = 100
learning_rate = 0.1
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
    
def error(predictions, labels):
    error_rate = 1
    errors = 0
    if (len(predictions) != len(labels)):
        print 'Different number of labels and predictions...'
    else:
        for x in range(len(predictions)):
            if predictions[x] != labels[x]:
                errors += 1
                
        error_rate = errors / len(predictions)
    return error_rate
    
def analyze(examples):
    report = ''
    report += 'Training examples: ' + str(len(examples)) + '\n'
    report += 'Features: ' + str(len(examples[0])) + '\n'
    
    #print ''
    #print 'Youngest:', min(np.transpose(examples)[0])
    #print 'Oldest:', max(np.transpose(examples)[0])
    #print ''
    
    report += 'Positives: ' + str(len([x for x in training if x[1] > 0])) + '\n'
    report += 'Negatives: ' + str(len([x for x in training if x[1] < 0])) + '\n'
    
    return report

if __name__ == '__main__':
    print 'Testing...a1.py'
    
    #a = np.arange(10) - 7
    #print a
    #print normalize(a)
    
    heart_data = np.genfromtxt(heart_file, delimiter=",", comments="#")
    #X = np.genfromtxt(gisette_file)

    #print heart_data
    #print len(heart_data)
    #print len(heart_data[0])

    num_for_testing = math.floor(len(heart_data) * reserve)
    
    #np.random.shuffle(heart_data)
    training = heart_data[num_for_testing:]
    testing = heart_data[:num_for_testing]
    
    print len(heart_data), 'examples loaded.'
    print len(training), 'for training'
    print len(testing), 'for testing'
    print analyze(training)
    
    training_labels, training_features = extract(training)
    testing_labels, testing_features = extract(testing)
    
    
    p = per.Perceptron(max_iterations, learning_rate)
    
    #normalized_training_features = [normalize(x) for x in training_features]
    p.fit(np.array(training_features), np.array(training_labels))
    
    print ''
    print 'WEIGHT VECTOR'
    print p.w
    print ''
    
    training_labels_predictions = [p.predict(i) for i in np.array(training_features)]
    print 'Training Data Error:', error(training_labels_predictions, training_labels)
    
    testing_labels_predictions = [p.predict(i) for i in np.array(testing_features)]
    print 'Testing Data Error:', error(testing_labels_predictions, testing_labels)


    b = per.PerceptronBias(max_iterations, learning_rate)
    
    
    b.fit(np.array(training_features), np.array(training_labels))
    
    print ''
    print 'WEIGHT VECTOR'
    print b.w
    print ''
    
    training_labels_predictions = [b.predict(i) for i in np.array(training_features)]
    print 'Training Data Error:', error(training_labels_predictions, training_labels)
    
    testing_labels_predictions = [b.predict(i) for i in np.array(testing_features)]
    print 'Testing Data Error:', error(testing_labels_predictions, testing_labels)