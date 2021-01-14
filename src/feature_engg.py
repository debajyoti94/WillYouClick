"""
Here we will pre-process the data and make it ready for the model to consume
"""

import config
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import abc
import pickle
import seaborn as sns
import sklearn.preprocessing as preproc


# creating a template of functions that i absolutely need
# so that i don't forget about it
class MustHaveForFeatureEngineering(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def cleaning_data(self, data):
        return

    @abc.abstractmethod
    def plot_null_values(self, data):
        # to check if any null values
        # are present in the dataset
        return


# inheriting from the abstract class
class FeatureEngineering(MustHaveForFeatureEngineering):

    # this class is responsible for
    # feature enginering on the  dataset
    def label_encoding(self, data, features_to_encode):
        """
        Using this function to encode categorical features
         (nominal to be precise)
        :param data: input data, whose feature you want to encode
        :param features_to_encode:
        :return: encoded features
        """
        le = LabelEncoder()
        encoded_feature = le.fit_transform(data[features_to_encode])

        return encoded_feature

    def feature_scaling(self, input_data, features_to_scale):
        """
        Applying minmax scaling here.
        :param features_to_scale:
        :return:
        """
        for feature in features_to_scale:
            input_data[str(feature) + "_scaled"] = preproc.minmax_scale(
                input_data[[feature]]
            )

        return input_data

    def cleaning_data(self, input_data):
        """
        All the data cleaning and feature engineering
         will happen here.
        :param input_data: data to clean
        :return: Cleaned up data
        """

        # drop the features that we do no want to keep
        # while training the model
        cleaned_input_data = input_data.drop(
            config.FEATURES_TO_DROP, axis=1, inplace=False
        )

        # label encoding feature: Country
        country_labels = self.label_encoding(
            cleaned_input_data, config.CATEGORICAL_VARIABLES
        )
        cleaned_input_data["Country_encoded"] = country_labels
        cleaned_input_data.drop("Country", axis=1, inplace=True)

        # apply feature scaling to salary and time spent on site
        cleaned_input_data = self.feature_scaling(
            cleaned_input_data, ["Salary", "Time Spent on Site"]
        )

        # drop the original features
        cleaned_input_data.drop(["Salary", "Time Spent on Site"],
                                            axis=1, inplace=True)

        # create the heatmap plot of null values as a check
        self.plot_null_values(cleaned_input_data)

        return cleaned_input_data


    def fill_fare(self, gender_fare_tuple):
        """
        Fare feature is missing, so we fill it up based
         on mean values obtained
        :param fare_gender_tuple: (sex,fare)
        :return: imputed fare
        """
        sex = gender_fare_tuple[0]
        fare = gender_fare_tuple[1]

        if pd.isnull(fare):
            if sex == "Male":
                return 8
            else:
                return 18
        else:
            return fare

    def plot_null_values(self, data):
        """
        Here we will make a heatmap plot to see if there are any null values
        :param data: input data
        :return: a heatmap, stored on disk
        """

        sns_heatmap_plot = sns.heatmap(data.isnull(), cmap="Blues", yticklabels=False)
        sns_heatmap_plot.figure.savefig(config.NULL_CHECK_HEATMAP)


class DumpLoadFile:
    def load_pickled_file(self, filename):
        """

        :param filename: file that you want to load
        :return: unpickled file
        """
        with open(filename, "rb") as pickle_handle:
            return pickle.load(pickle_handle)

    def dump_file(self, data, filename):
        """

        :param data:  data that we want to dump/serialize
        :param filename: filename that we want to associate to the data
        :return: 0 if it works out well, -1 if in case something fails
        """

        with open(filename, "wb") as pickle_handle:
            pickle.dump(data, pickle_handle)

