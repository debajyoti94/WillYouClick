''' This file will contain the variables that will be used for training or other purposes'''

# dataset information
OUTPUT_FEATURE = 'Clicked'
FEATURES_TO_DROP = ['Names','emails']
CATEGORICAL_VARIABLES = ['Country']
TRAIN_FILENAME  = 'train_set.pickle'
TEST_FILENAME = 'test_set.pickle'

# for cross validation
NUM_FOLDS = 5

# model name
MODEL_NAME = 'LR_Ad_click_Baseline_'

FILE_DELIMITER = ','
ENCODING_TYPE = 'ISO-8859-1'
DATASET_SHAPE = (499,6)
