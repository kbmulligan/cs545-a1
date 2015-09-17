# a1.py - 
# by K. Brett Mulligan
# CSU - CS545 (Fall 2015)
# Assignment 1

# Note: Write custom sign function


from __future__ import division

import math
import time
import numpy as np
from matplotlib import pyplot as plt


import perceptron as per

max_iterations = 1000
learning_rate = 0.2
reserve = 0.2      # percent of examples to reserve for testing

heart_file = "heart.csv"
gisette_data_file = "gisette_train.data"
gisette_label_file = "gisette_train.labels"

def normalize(a):
    """Returns normalized 1-dimensional vector z from given vector a. 
       Finds min/max range in vector a and divides each member of a by that value."""
    # all abs
    b = [abs(x) for x in a]
    largest = max(b)
    z = [x/largest for x in a]
    # z = [(2*(x - min(a))/(max(a) - min(a))) - 1 for x in a]
    return z

def extract(examples):
    ids = [patient[0] for patient in examples]
    labels = [patient[1] for patient in examples]
    raw_data = [patient[2:] for patient in examples]
    
    return labels, raw_data
    
def analyze(examples):
    report = ''
    report += 'Training examples: ' + str(len(examples)) + '\n'
    report += 'Features: ' + str(len(examples[0])) + '\n'
    
    #print ''
    #print 'Youngest:', min(np.transpose(examples)[0])
    #print 'Oldest:', max(np.transpose(examples)[0])
    #print ''
    
    report += 'Positives: ' + str(len([x for x in examples if x[1] > 0])) + '\n'
    report += 'Negatives: ' + str(len([x for x in examples if x[1] < 0])) + '\n'
    
    return report


def test_perceptron(algorithm, training_data, testing_data):

    start = time.time()

    training_labels, training_features = extract(training_data)
    testing_labels, testing_features = extract(testing_data)

    print 'Testing', algorithm.__name__, '...'
    model = algorithm(max_iterations, learning_rate)

    model.fit(np.array(training_features), np.array(training_labels))
    
    # print ''
    # print 'WEIGHT VECTOR'
    # print model.w
    # print ''
    
    training_labels_predictions = [model.predict(i) for i in np.array(training_features)]
    print 'Training Data Error (E_in):', model.error(training_labels_predictions, training_labels)
    
    testing_labels_predictions = [model.predict(i) for i in np.array(testing_features)]
    print 'Testing Data Error (E_out):', model.error(testing_labels_predictions, testing_labels)

    

    end = time.time()

    print 'Elapsed time: ', end - start
    print ''
    return

def do_learning_curve(algorithm, training_data, testing_data):

    start = time.time()

    print '\n'
    print '--------- LEARNING CURVES -----------'
    print ''

    training_labels, training_features = extract(training_data)
    testing_labels, testing_features = extract(testing_data)

    print 'Testing learning curve using', algorithm.__name__, '...'
    model = algorithm(max_iterations, learning_rate)

    max_num_samples = len(training_data)

    E_in = []
    E_out = []

    step = 10
    samples = 100

    # x_range = np.arange(1, max_num_samples, step)
    x_range_floats = np.logspace(0, 3.6, samples)
    # print x_range_floats

    x_range = [int(i) for i in np.floor(x_range_floats)]
    # print x_range


    for x in x_range:
        model.fit(np.array(training_features[:x]), np.array(training_labels[:x]))

        training_labels_predictions = [model.predict(i) for i in np.array(training_features)]
        testing_labels_predictions = [model.predict(i) for i in np.array(testing_features)]
        E_in.append(model.error(training_labels_predictions, training_labels))
        E_out.append(model.error(testing_labels_predictions, testing_labels))


    # Error rate
    y = E_in

    # plot it
    plt.semilogx(x_range, y)


    # plot setup
    plt.xlabel('Training Samples (x)')
    plt.ylabel('In-sample error (E_in)')
    plt.title('Learning Curve')
    plt.grid(True)
    plt.savefig("learning_curve_Ein.png")
    plt.show()

    # Error rate
    y = E_out

    # plot it
    plt.semilogx(x_range, y)


    # plot setup
    plt.xlabel('Training Samples (x)')
    plt.ylabel('Out-of-sample error (E_out)')
    plt.title('Learning Curve')
    plt.grid(True)
    plt.savefig("learning_curve_Eout.png")
    plt.show()
    

    end = time.time()

    print 'Elapsed time: ', end - start
    print ''
    return


if __name__ == '__main__':
    print 'Testing...a1.py'

    print ''
    print 'Max iterations: ', max_iterations
    print 'learning_rate: ', learning_rate
    print ''


    ####### HEART DATA ###########
    print '--------- HEART DATA -----------'
    print 'Loading data...'

    heart_data = np.genfromtxt(heart_file, delimiter=",", comments="#")

    # h_num_for_testing = int(math.floor(len(heart_data) * reserve))
    h_num_for_testing = 100
    
    np.random.shuffle(heart_data)
    htraining = heart_data[h_num_for_testing:]
    htesting = heart_data[:h_num_for_testing]
    
    
    print len(heart_data), 'examples loaded.'
    print len(htraining), 'for training'
    print len(htesting), 'for testing'
    print ''

    print analyze(htraining)

    print 'Results for UN-SCALED data:\n'
    
    test_perceptron(per.Perceptron, htraining, htesting)
    test_perceptron(per.PerceptronBias, htraining, htesting)
    test_perceptron(per.PerceptronPocket, htraining, htesting)
    test_perceptron(per.PerceptronModified, htraining,htesting)



    ####### GISETTE DATA ###########

    print '\n'
    print '--------- GISETTE DATA -----------'
    print 'Loading data...'

    gisette_data_features = np.genfromtxt(gisette_data_file)
    gisette_labels = np.genfromtxt(gisette_label_file)

    gisette_data = []
    for i in range(len(gisette_data_features)):
        gisette_data.append(np.concatenate(([0], [gisette_labels[i]], gisette_data_features[i])))

    # # print len(gisette_data)
    # # print len(gisette_data[0])

    
    # # g_num_for_testing = int(math.floor(len(gisette_data) * reserve))
    g_num_for_testing = 1500

    np.random.shuffle(gisette_data)
    gtraining = gisette_data[g_num_for_testing:]
    gtesting = gisette_data[:g_num_for_testing]


    print len(gisette_data), 'examples loaded.'
    print len(gtraining), 'for training'
    print len(gtesting), 'for testing'
    print ''

    test_perceptron(per.Perceptron, gtraining, gtesting)
    test_perceptron(per.PerceptronBias, gtraining, gtesting)
    test_perceptron(per.PerceptronPocket, gtraining, gtesting)
    test_perceptron(per.PerceptronModified, gtraining, gtesting)



    ####### LEARNING CURVES ###########

    # do_learning_curve(per.PerceptronBias, gtraining, gtesting)


    ####### NORMALIZATION ###########

    transposed_heart_data = np.transpose(heart_data)
    normalized_heart_data_by_feature = [normalize(feature) for feature in transposed_heart_data]
    normalized_heart_data = np.transpose(normalized_heart_data_by_feature)

    print ''
    print 'Un-normalized'
    print heart_data
    
    print ''
    print 'Normalized'
    print normalized_heart_data

    print heart_data[0]
    print normalized_heart_data[0]

    print '--------- NORMALIZED HEART DATA -----------'


    hntraining = normalized_heart_data[h_num_for_testing:]
    hntesting = normalized_heart_data[:h_num_for_testing]
    
    
    print len(heart_data), 'examples loaded.'
    print len(htraining), 'for training'
    print len(htesting), 'for testing'
    print ''

    print analyze(htraining)

    print 'Results for SCALED data:\n'
    
    test_perceptron(per.Perceptron, hntraining, hntesting)
    test_perceptron(per.PerceptronBias, hntraining, hntesting)
    test_perceptron(per.PerceptronPocket, hntraining, hntesting)
    test_perceptron(per.PerceptronModified, hntraining,hntesting)