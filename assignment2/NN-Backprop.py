"""
Backprop NN training
"""
# Adapted from https://github.com/JonathanTay/CS-7641-assignment-2/blob/master/NN0.py

import sys

sys.path.append("./ABAGAIL.jar")
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from func.nn.backprop import RPROPUpdateRule, BatchBackPropagationTrainer
from func.nn.activation import RELU
from base import *

# Network parameters found "optimal" in Assignment 1
INPUT_LAYER = 7
HIDDEN_LAYER1 = 16
HIDDEN_LAYER2 = 16
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 5001
OUTFILE = OUTPUT_DIRECTORY + '/NN_OUTPUT/NN_{}_LOG.csv'


def main():
    """Run this experiment"""
    training_ints = initialize_instances(TRAIN_DATA_FILE)
    testing_ints = initialize_instances(TEST_DATA_FILE)
    validation_ints = initialize_instances(VALIDATE_DATA_FILE)
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(training_ints)
    relu = RELU()
    # 50 and 0.000001 are the defaults from RPROPUpdateRule.java
    rule = RPROPUpdateRule(0.064, 50, 0.000001)
    oa_names = ["Backprop"]
    classification_network = factory.createClassificationNetwork(
        [INPUT_LAYER, HIDDEN_LAYER1, HIDDEN_LAYER2, OUTPUT_LAYER], relu)
    train(BatchBackPropagationTrainer(data_set, classification_network, measure, rule), classification_network,
          'Backprop', training_ints, validation_ints, testing_ints, measure, TRAINING_ITERATIONS,
          OUTFILE.format('Backprop'))


if __name__ == "__main__":
    with open(OUTFILE.format('Backprop'), 'a+') as f:
        f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format('iteration', 'MSE_trg', 'MSE_val', 'MSE_tst', 'acc_trg',
                                                            'acc_val', 'acc_tst', 'f1_trg', 'f1_val', 'f1_tst',
                                                            'elapsed'))
    main()
