''' This file will contain the variables that will be used for training or other purposes'''

# dataset information
OUTPUT_FEATURE = 'Clicked'
FEATURES_TO_DROP = ['Names','emails']
CATEGORICAL_VARIABLES = ['Country']
ORIGINAL_DATASET_FILENAME = '../input/Facebook_Ads_2.csv'
TRAIN_FILENAME  = '../input/train_set.pickle'
TEST_FILENAME = '../input/test_set.pickle'

# for plots
NULL_CHECK_HEATMAP = '../plots/null_check_heatmap.png'

# for cross validation
NUM_FOLDS = 5

# model name
MODEL_NAME = '../models/LR_Ad_click_Baseline_'

FILE_DELIMITER = ','
ENCODING_TYPE = 'ISO-8859-1'
DATASET_SHAPE = (499,6)
