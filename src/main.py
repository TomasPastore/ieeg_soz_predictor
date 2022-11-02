import os
import sys
import time
import warnings
from ml_hfo_classifier import boost_train
from utils import get_patient_data

warnings.filterwarnings("ignore", module="matplotlib")
warnings.simplefilter(action='ignore', category=FutureWarning)
from conf import FIG_FOLDER_DIR, FIG_SAVE_PATH
from db_parsing import Database, get_granularity, HFO_TYPES
from driver import Driver
import argparse
import profiling


# from profiling import profile_memory

def main(interactive_exp_menu=False):
    db = Database()
    elec_collection, evt_collection = db.get_collections()
    exp_driver = Driver(elec_collection, evt_collection)

    # Paper Frontiers
    # phase_coupling_paper(hfo_collection) # Paper Frontiers

    # Thesis
    if interactive_exp_menu:
        experiment_menu(exp_driver)
    else:
        # get_patient_data(elec_collection, evt_collection,
        # output_csv=FIG_FOLDER_DIR)
        # boost_train('Hippocampus', 'Fast RonO', FIG_SAVE_PATH[4]['ii'][
        #    'Hippocampus'])

        # Call an specific driver function if not interactive mode
        for location in ['Whole Brain', 'Limbic Lobe', 'Frontal Lobe',
                         'Hippocampus']:
            for hfo_type in HFO_TYPES:
                ml_args = define_ml_args(hfo_type, location)
                exp_driver.run_experiment(number=4, **ml_args)


def define_ml_args(hfo_type, location):
    return {
        'evt_types_to_load': [hfo_type],
        'locations': {get_granularity(location): [location]}
    }


def choose_ml_hfo_type(exp_driver, location):
    clear_screen()
    print('\nPick HFO type:')
    print('                ')
    print('\t1) RonO')
    print('\t2) RonS')
    print('\t3) Fast RonO')
    print('\t4) Fast RonS')
    print('\t5) Go Back')
    option = int(input('\nChoose a number from the options above: '))
    if option == 1:
        hfo_type = 'RonO'
        ml_args = define_ml_args(hfo_type, location)
        exp_driver.run_experiment(number=4, **ml_args)
        go_to_menu_after(5, exp_driver)
    elif option == 2:
        hfo_type = 'RonS'
        ml_args = define_ml_args(hfo_type, location)
        exp_driver.run_experiment(number=4, **ml_args)
        go_to_menu_after(5, exp_driver)
    elif option == 3:
        hfo_type = 'Fast RonO'
        ml_args = define_ml_args(hfo_type, location)
        exp_driver.run_experiment(number=4, **ml_args)
        go_to_menu_after(5, exp_driver)
    elif option == 4:
        hfo_type = 'Fast RonS'
        ml_args = define_ml_args(hfo_type, location)
        exp_driver.run_experiment(number=4, **ml_args)
        go_to_menu_after(5, exp_driver)
    elif option == 5:
        choose_ml_location(exp_driver)


def choose_ml_location(exp_driver):
    clear_screen()
    print('\nPick Location:')
    print('                ')
    print('\t1) Whole Brain')
    print('\t2) Limbic Lobe')
    print('\t3) Frontal Lobe')
    print('\t4) Hippocampus')
    print('\t5) Go Back')
    option = int(input('\nChoose a number from the options above: '))
    if option == 1:
        location = 'Whole Brain'
        choose_ml_hfo_type(exp_driver, location)
    elif option == 2:
        location = 'Limbic Lobe'
        choose_ml_hfo_type(exp_driver, location)
    elif option == 3:
        location = 'Frontal Lobe'
        choose_ml_hfo_type(exp_driver, location)
    elif option == 4:
        location = 'Hippocampus'
        choose_ml_hfo_type(exp_driver, location)
    elif option == 5:
        go_to_menu_after(5, exp_driver)


def experiment_menu(exp_driver):
    clear_screen()
    print('\nMain functions:')
    print('                ')
    print('\t1) Data dimensions')
    print('\t2) Data stats analysis. Features and HFO rate in SOZ vs NSOZ.')
    print('\t3) Predicting SOZ with event rates: Baselines')
    print('\t4) ML HFO classifiers')
    print('\t5) Exit')
    option = int(input('\nChoose a number from the options above: '))

    if option == 1:
        exp_driver.run_experiment(number=1)
        go_to_menu_after(5, exp_driver)
    elif option == 2:
        exp_driver.run_experiment(number=2, roman_num='ii')  # localized
        go_to_menu_after(5, exp_driver)
    elif option == 3:
        exp_driver.run_experiment(number=3)
        go_to_menu_after(5, exp_driver)
    elif option == 4:
        choose_ml_location(exp_driver)
    elif option == 5:
        sys.exit()
    else:
        raise NotImplementedError('Option {0} was left as future '
                                  'work.'.format(option))
        go_to_menu_after(5, exp_driver)


def clear_screen():
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')


def go_to_menu_after(seconds, exp_driver):
    while seconds > 0:
        print('Going back to menu in {0}...'.format(seconds))
        time.sleep(1)  # wait 1 sec
        seconds = seconds - 1
    experiment_menu(exp_driver)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interactive_mode",
                        help="Run the experiments interactively.",
                        required=False,
                        default=False,
                        action='store_true',
                        )
    args = parser.parse_args()
    # with profiling.timer_for('IEEG SOZ Predictor') as timer:
    main(interactive_exp_menu=args.interactive_mode)
