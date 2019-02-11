import csv
import os
import time

import random as rand
from shared import Instance
from java.lang import Math

__all__ = ['DS_NAME', 'TEST_DATA_FILE', 'TRAIN_DATA_FILE', 'VALIDATE_DATA_FILE', 'OUTPUT_DIRECTORY',
           'initialize_instances', 'error_on_data_set', 'train']

DS_NAME = 'HTRU2Data'

TEST_DATA_FILE = 'data/{}_test.csv'.format(DS_NAME)
TRAIN_DATA_FILE = 'data/{}_train.csv'.format(DS_NAME)
VALIDATE_DATA_FILE = 'data/{}_validate.csv'.format(DS_NAME)

# TODO: Move this to a common lib?
OUTPUT_DIRECTORY = './output'

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
if not os.path.exists(OUTPUT_DIRECTORY + '/images'):
    os.makedirs(OUTPUT_DIRECTORY + '/images')
subdirs = ['NN_OUTPUT', 'CONTPEAKS', 'FLIPFLOP', 'TSP']
for subdir in subdirs:
    if not os.path.exists('{}/{}'.format(OUTPUT_DIRECTORY, subdir)):
        os.makedirs('{}/{}'.format(OUTPUT_DIRECTORY, subdir))
    if not os.path.exists('{}/images/{}'.format(OUTPUT_DIRECTORY, subdir)):
        os.makedirs('{}/images/{}'.format(OUTPUT_DIRECTORY, subdir))

seed = 653091685
# seed = rand.randint(0, (2 ** 32) - 1)
print("Using seed {}".format(seed))
rand.seed(seed)


def initialize_instances(infile):
    """Read the given CSV data into a list of instances."""
    instances = []

    # Read in the CSV file
    with open(infile, "r") as dat:
        reader = csv.reader(dat)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            # TODO: Set to <= 0 to handle 0/1 labels and not just -1/1?
            instance.setLabel(Instance(0 if float(row[-1]) < 0 else 1))
            instances.append(instance)

    return instances


# Adapted from:
# https://codereview.stackexchange.com/questions/36096/implementing-f1-score
# https://www.kaggle.com/hongweizhang/how-to-calculate-f1-score
# https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/
def f1_score(labels, predicted):
    get_count = lambda x: sum([1 for i in x if i is True])

    tp = get_count([predicted[i] == x and x == 1.0 for i, x in enumerate(labels)])
    tn = get_count([predicted[i] == x and x == 0.0 for i, x in enumerate(labels)])
    fp = get_count([predicted[i] == 1.0 and x == 0.0 for i, x in enumerate(labels)])
    fn = get_count([predicted[i] == 0.0 and x == 1.0 for i, x in enumerate(labels)])

    if tp == 0:
        return 0, 0, 0

    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return precision, recall, 0.0
    return precision, recall, f1


def error_on_data_set(network, ds, measure, ugh=False):
    N = len(ds)
    error = 0.
    correct = 0
    incorrect = 0
    actuals = []
    predicteds = []
    for instance in ds:
        network.setInputValues(instance.getData())
        network.run()
        actual = instance.getLabel().getContinuous()
        predicted = network.getOutputValues().get(0)
        predicted = max(min(predicted, 1), 0)
        if ugh:
            print "label: {}".format(instance.getLabel())
            print "actual: {}, predicted: {}".format(actual, predicted)

        predicteds.append(round(predicted))
        actuals.append(max(min(actual, 1), 0))
        if abs(predicted - actual) < 0.5:
            correct += 1
            if ugh:
                print "CORRECT"
        else:
            incorrect += 1
            if ugh:
                print "INCORRECT"
        output = instance.getLabel()
        output_values = network.getOutputValues()
        example = Instance(output_values, Instance(output_values.get(0)))
        error += measure.value(output, example)
        if ugh:
            print "error: {}".format(measure.value(output, example))

    MSE = error / float(N)
    acc = correct / float(correct + incorrect)
    precision, recall, f1 = f1_score(actuals, predicteds)
    if ugh:
        print "MSE: {}, acc: {}, f1: {} (precision: {}, recall: {})".format(MSE, acc, f1, precision, recall)
        import sys
        sys.exit(0)

    return MSE, acc, f1


def train(oa, network, oaName, training_ints, validation_ints, testing_ints, measure, training_iterations, outfile):
    """Train a given network on a set of instances.
    """
    print "\nError results for %s\n---------------------------" % (oaName,)
    times = [0]
    for iteration in xrange(training_iterations):
        start = time.clock()
        oa.train()
        elapsed = time.clock() - start
        times.append(times[-1] + elapsed)
        if iteration % 10 == 0:
            MSE_trg, acc_trg, f1_trg = error_on_data_set(network, training_ints, measure)
            MSE_val, acc_val, f1_val = error_on_data_set(network, validation_ints, measure)
            MSE_tst, acc_tst, f1_tst = error_on_data_set(network, testing_ints, measure)
            txt = '{},{},{},{},{},{},{},{},{},{},{}\n'.format(iteration, MSE_trg, MSE_val, MSE_tst, acc_trg, acc_val,
                                                             acc_tst, f1_trg, f1_val, f1_tst, times[-1])
            print txt
            with open(outfile, 'a+') as f:
                f.write(txt)
