""" In this code we will put down the unit test cases """

import config
import pandas as pd
from feature_engg import DumpLoadFile
import csv


class TestFunctionalities:
    def column_null_check(self):
        """
        Function to check if there are any missing
        values in the columns
        :return: True if any missing values
        """

        # load dataset
        dl_obj = DumpLoadFile()
        train_dataset = dl_obj.load_file(config.TRAIN_FILENAME)
        test_dataset = dl_obj.load_file(config.TEST_FILENAME)

        train_null_col_arr = pd.isnull(train_dataset)
        test_null_coll_arr = pd.isnull(test_dataset)

        assert False in train_null_col_arr or False in test_null_coll_arr

    def delimiter_check(self):
        """
        Check if the delimiter same as what is provided in config file
        :return: True if all is okay
        """
        with open(config.ORIGINAL_FILENAME, "r") as csv_file:
            file_contents = csv.Sniffer().sniff(csv_file.readline())

        assert True if file_contents.delimiter == config.FILE_DELIMITER \
                                                            else False

    def test_dataset_shape(self):
        """
        Check if the instances x features match with be
        what is expected
        :return: True if all is okay
        """
        dataset = pd.read_csv(
            config.ORIGINAL_DATASET_FILENAME,
            delimiter=config.FILE_DELIMITER,
            encoding=config.ENCODING_TYPE,
        )

        assert True if dataset.shape == config.DATASET_SHAPE else False
