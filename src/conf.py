# Global variables and settings

from pathlib import Path
import utils
import ml_algorithms

DEBUG = True
TEST_BEFORE_RUN = False

ML_MODELS_TO_RUN = ['SGD']  # Default ml_algorithms

CLASSIFIERS = ml_algorithms.CLASSIFIERS

PROJECT_ROOT_DIR = utils.get_project_root()
VALIDATION_NAMES_BY_LOC_PATH = str(Path(PROJECT_ROOT_DIR,
                                        'src/validation_names_by_loc.json'))

# ORCA dependency is required if you want automatically save plotly figs In
# Ubuntu 20 I installed with sudo npm install -g --verbose --unsafe-perm=true
# --allow-root electron@6.1.4 orca If this path below doesnt exist code will
# skip figures saves, type 'which orca' in shell to get it after installing
# ORCA_EXECUTABLE = '/home/tpastore/.npm/versions/node/v14.5.0/bin/orca'  #
# tpastore HP pavilion 15
ORCA_EXECUTABLE = '/usr/local/bin/orca'  # tpastore Dell

FIG_FOLDER_DIR = str(Path(PROJECT_ROOT_DIR, 'figures'))

# Paths for saving figures for each experiment defined in driver.py
# Note: untagged versions are deprecated
FIG_SAVE_PATH = dict()

# 1 Data dimensions and sleep patients
FIG_SAVE_PATH[1] = str(Path(FIG_FOLDER_DIR, '1_data_dimensions'))  # dir

# 2 Stats of features and event rate SOZ vs NSOZ
FIG_SAVE_PATH[2] = dict()
FIG_SAVE_PATH[2]['dir'] = str(Path(FIG_FOLDER_DIR, '2_stats'))

# 3 Event rate soz predictor baselines
FIG_SAVE_PATH[3] = dict()
FIG_SAVE_PATH[3]['dir'] = str(
    Path(FIG_FOLDER_DIR, '3_rate_soz_predictor_baselines'))

# 4 ml_hfo_classfiers
FIG_SAVE_PATH[4] = dict()
FIG_SAVE_PATH[4]['dir'] = str(Path(FIG_FOLDER_DIR, '4_ml_hfo_classifiers'))
