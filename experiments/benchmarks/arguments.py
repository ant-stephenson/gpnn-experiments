import argparse


def _shared_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--datafile', help="location of '.npy' data file (required). Note: last column (-1) "
                                                 'will be interpreted as the variable being predicted, all other '
                                                 'columns will be used as input variables', required=True)
    parser.add_argument('-P', '--preprocess-method',
                        help='mechanism by which to pre-process our data (default: scaling)',
                        choices=['scaling', 'whitening', 'none'], default='scaling')
    parser.add_argument('-p', '--train-prop', help='proportion to use as training data (default: 7/9)', default=7/9,
                        type=float)
    parser.add_argument('-s', '--tt-seed', help='seed to use in random train-test splitting (default: None)',
                        default=None, type=int)
    parser.add_argument('-i', '--iters', help='number of iterations/epochs for training optimiser (default: 100)',
                        default=100, type=int)
    parser.add_argument('-l', '--lr', help='learning rate for training optimiser (default: 0.01)', default=0.01,
                        type=float)
    parser.add_argument('-b', '--batch-size', help='batch size for prediction (default: 1000)', default=1000,
                        type=int)
    parser.add_argument('-o', '--output', help='file name for output results (required)', required=True)
    return parser


def parse_distributed_args():
    parser = _shared_args()
    parser.add_argument('-S', '--partition-seed', help='seed to use in random expert partitioning (default: None)',
                        default=None, type=int)
    parser.add_argument('-m', '--points-per-expert',
                        help='desired number of points per expert (approximate; default: 625)', default=625, type=int)
    parser.add_argument('-M', '--model-type', help='which models to run (default: all)',
                        choices=['all', 'bcm', 'rbcm', 'grbcm', 'poe', 'gpoe'],
                        default='all')
    args = parser.parse_args()
    return args


def parse_variational_args():
    parser = _shared_args()
    parser.add_argument('-I', '--inducing-points', help='Number of inducing points to use (default: 1024)',
                        default=1024, type=int)
    parser.add_argument('-m', '--minibatch-training-size', help='minibatch size for training (default: 1024)',
                        default=1024, type=int)
    return parser.parse_args()
