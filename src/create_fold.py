""" Here we will write the code for
 stratified k fold cross validation"""

import config
from sklearn import model_selection


class SKFold:
    def create_folds(self, dataset_df):
        """
        This function creates the stratified kfolds
        :return: dataset with kfold column containing the values
        """

        kf = model_selection.StratifiedKFold(
            n_splits=config.NUM_FOLDS, shuffle=True, random_state=0
        )

        # fetch target class
        y = dataset_df[config.OUTPUT_FEATURE].values

        for fold_value, (t_, y_index) in enumerate(kf.split(X=dataset_df, y=y)):
            dataset_df.loc[y_index, "kfold"] = fold_value

        return dataset_df
