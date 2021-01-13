'''
Here we will write code for training the model.
This code will code contain things associated to adding commandline arguments.
'''


import argparse
import config
from create_fold import SKFold
from feature_engg import FeatureEngineering, \
                            DumpLoadFile
import pandas as pd
import os
from sklearn import linear_model
from sklearn.metrics import accuracy_score,\
                        precision_recall_fscore_support

def run(fold, dataset):
    '''
    We will train the model here
    :param fold:
    :param dataset:
    :return:
    '''

    # split the train set into X_train and y_train
    dataset_train = dataset[dataset.kfold != fold].reset_index(drop=True)
    dataset_valid = dataset[dataset.kfold == fold].reset_index(drop=True)

    X_train = dataset_train.drop([config.OUTPUT_FEATURE, 'kfold'], axis=1,
                           inplace=False).values
    y_train = dataset_train[config.OUTPUT_FEATURE].values

    X_valid = dataset_valid.drop([config.OUTPUT_FEATURE, 'kfold'], axis=1,
                                 inplace=False).values
    y_valid = dataset_valid[config.OUTPUT_FEATURE].values

    lr_model = linear_model.LogisticRegression(penalty='l2',
                                               class_weight='balanced')

    # train model
    lr_model.fit(X_train, y_train)

    # get metrics
    preds = lr_model.predict(X_valid)

    accuracy = accuracy_score(y_valid, preds)
    p, r, f1, support = precision_recall_fscore_support(y_valid, preds)

    print(
        "---Fold={}---\nAccuracy={}\nPrecision={}\nRecall={}\nF1={}".format(
            fold, accuracy, p, r, f1
        )
    )

    dl_obj = DumpLoadFile()
    dl_obj.dump_file(lr_model, str(config.MODEL_NAME)+str(fold)+'.pickle')


def inference_stage(model, dataset):
    '''

    :param model:
    :param dataset: test set preferrably
    :return:
    '''

    X_test = dataset.drop(config.OUTPUT_FEATURE, axis=1,
                                                inplace=False).values
    y_test = dataset[config.OUTPUT_FEATURE].values

    # get predictions
    preds = model.predict(X_test)

    # get performance metrics
    accuracy = accuracy_score(y_test, preds)

    p, r, f1, support = precision_recall_fscore_support(y_test, preds)

    print(
        "---Test set performance---\nAccuracy={}\nPrecision={}\nRecall={}\nF1={}".format(
             accuracy, p, r, f1
        )
    )

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
                             ' on the test set.')

    args = parser.parse_args()
    dl_obj = DumpLoadFile()     # this is for pickling objects

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
        shuffled_data = clean_data.sample(frac=1,
                                          random_state=0).reset_index(
                                                                    drop=True)

        # split into train and test
        train_set = shuffled_data[:400]
        test_set = shuffled_data[400:]

        # dump the dataset
        dl_obj.dump_file(train_set, config.TRAIN_FILENAME)
        dl_obj.dump_file(test_set, config.TEST_FILENAME)

    elif args.train == 'skfold':

        # first check if the train set exists
        # if it doesn't then we need to get the training dataset first
        if os.path.isfile(config.TRAIN_FILENAME):
            # load the train set
            train_set = dl_obj.load_pickled_file(config.TRAIN_FILENAME)

            # get the skfold
            train_set['kfold'] = -1
            skfold_obj = SKFold()
            train_set = skfold_obj.create_folds(train_set)

            #train the model
            for fold in range(config.NUM_FOLDS):
                run(fold,train_set)

        else:
            print("Training set does not exist. Please obtain the train set first.\n"
                  "Use \"python train.py --clean dataset\" to get the train and test set.")

    elif args.test == 'inference':

        # first check if the test set exists
        if os.path.isfile(config.TEST_FILENAME):
            # load the test set
            test_set = dl_obj.load_pickled_file(config.TEST_FILENAME)

            # load the model
            model = dl_obj.load_pickled_file(config.BEST_MODEL)
            # call the inference stage
            inference_stage(model, test_set)

        else:
            print("Test set does not exist. Please obtain the test set first.\n"
                  "Use \"python train.py --clean dataset\" to get the train and test set.")