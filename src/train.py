'''
Here we will write code for training the model.
This code will code contain things associated to adding commandline arguments.
'''


import argparse
import config
import create_folds
from feature_engg import FeaturEngineering
import pandas as pd
import os

def run():
    '''
    We will train the model here
    :return:
    '''

    # train model

    # get metrics
    return


def inference_stage():
    '''
    Get the predictions for test set here
    :return:
    '''
    # load model

    # get predictions

    # get performance metrics

    return

if __name__  == "__main__":

    # code for argparse comes here
    parser = argparse.ArgumentParser()

    parser.add_argument('--clean', type=str,
                        help='Provide argument \"--clean dataset\" to get'
                             ' clean train and test split.')

    parser.add_argument('--train', type=str,
                        help='Provide argument \"--train skfold\" to train'
                             ' model using Stratified KFold cross validation.')

    parser.add_argument('--test', type=str,
                        help='Provide argument \"--test inference\" to get model performance'
                             ' on the test set')

    args = parser.parse_args()
    # based on commandline arguments
    # call train / inference stage functions
    if args.clean == 'dataset':
        # this is where we will call the clean data function
        # and split the data into train and test

        # get the original dataset
        original_dataset = pd.read_csv(config.ORIGINAL_DATASET_FILENAME,
                                       delimiter=config.FILE_DELIMITER,
                                       encoding=config.ENCODING_TYPE)

        # get clean data
        fr_obj = FeatureEngineering()
        clean_data = fr_obj.cleaning_data(original_dataset)

        # shuffle the cleaned data
        shuffled_data = clean_data.sample(frac=1).reset_index(drop=True)

        # split into train and test
        train_set = shuffled_data[:400]
        test_set = shuffled_data[400:]

        # dump the dataset
        fr_obj.dump_file(train_set, config.TRAIN_FILENAME)
        fr_obj.dump_file(test_set, config.TEST_FILENAME)

    elif args.train == 'skfold':

        # first check if the train set exists
        # if it doesn't then we need to get the training dataset first
        if os.path.isfile(config.TRAIN_FILENAME):
            #train the model

        else:
            print("Training set does not exist. Please obtain the train set first.\n"
                  "Use \"python train.py --clean dataset\" to get the train and test set.")

    elif args.test == 'inference':

        # first check if the test set exists
        if os.path.isfile(config.TEST_FILENAME):
            # call the inference stage

        else:
            print("Test set does not exist. Please obtain the test set first.\n"
                  "Use \"python train.py --clean dataset\" to get the train and test set.")