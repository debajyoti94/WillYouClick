''' This file will contain the variables that will be used for training or other purposes'''

# for feature engineering
OUTPUT_FEATURE = 'Clicked'
FEATURES_TO_DROP = ['Names','emails']

# for cross validation
NUM_FOLDS = 5

# model name
MODEL_NAME = 'LR_Baseline_'

FILE_DELIMITER = ','
ENCODING_TYPE = 'ISO-8859-1'
DATASET_SHAPE = (499,6)