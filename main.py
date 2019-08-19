import os
# import pickle
import sys
import numpy as np
from time import time

import time as time0
from datetime import datetime

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from dataset import DataSet

parser = ArgumentParser("deep_personalization",
						  formatter_class=ArgumentDefaultsHelpFormatter,
						  conflict_handler='resolve')

parser.add_argument('--num_epoch', default=1, type=int,
					  help='The number of the epochs.')

parser.add_argument('--batch_size', default=32, type=int,
					  help='Batch size of the input data.')

args = parser.parse_args()


def train_network():
    for idx, epoch in enumerate(datasource.gen_epochs()):
        for X_train,X_train_iden,Y_train,X_valid,X_valid_iden,Y_valid in epoch:
            print(np.array(Y_train).shape,np.array(X_train).shape,np.array(X_train_iden).shape,)
def main():
    datasource = DataSet(batch_size=args.batch_size, num_epoch=args.num_epoch)

    datasource.prepare_dataset()



if __name__ == '__main__':
    main()