"""
GA NN training
"""
# Adapted from https://github.com/JonathanTay/CS-7641-assignment-2/blob/master/NN3.py
import sys

sys.path.append("./ABAGAIL.jar")
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem
from func.nn.backprop import RPROPUpdateRule
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
from func.nn.activation import RELU
from base import *

# Network parameters found "optimal" in Assignment 1
INPUT_LAYER = 8
HIDDEN_LAYER1 = 16
HIDDEN_LAYER2 = 16
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 5001
OUTFILE = OUTPUT_DIRECTORY + '/NN_OUTPUT/NN_{}_LOG.csv'


def main(P, mate, mutate):
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
    oa_name = "GA_{}_{}_{}".format(P, mate, mutate)
    with open(OUTFILE.format(oa_name), 'w') as f:
        f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format('iteration', 'MSE_trg', 'MSE_val', 'MSE_tst', 'acc_trg',
                                                            'acc_val', 'acc_tst', 'f1_trg', 'f1_val', 'f1_tst',
                                                            'elapsed'))
    classification_network = factory.createClassificationNetwork(
        [INPUT_LAYER, HIDDEN_LAYER1, HIDDEN_LAYER2, OUTPUT_LAYER], relu)
    nnop = NeuralNetworkOptimizationProblem(data_set, classification_network, measure)
    oa = StandardGeneticAlgorithm(P, mate, mutate, nnop)
    train(oa, classification_network, oa_name, training_ints, validation_ints, testing_ints, measure,
          TRAINING_ITERATIONS, OUTFILE.format(oa_name))


if __name__ == "__main__":
    for p in [50]:
        for mate in [20, 10]:
            for mutate in [20, 10]:
                args = (p, mate, mutate)
                main(*args)
