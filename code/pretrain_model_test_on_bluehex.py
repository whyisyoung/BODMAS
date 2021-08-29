'''
/*
 * @Author: Limin Yang (liminy2@illinois.edu)
 * @Date: 2021-08-26 15:42:50
 * @Last Modified by: Limin Yang
 * @Last Modified time: 2021-08-29 02:33:00
 * NOTE: script for Table II (Sophos-DNN was not included).
 */
'''


# Sophos DNN result was in /home/liminyang/github/SOREL-20M, not included in this repo.

import os

os.environ['PYTHONHASHSEED'] = '0'
from numpy.random import seed
import random
random.seed(1)
seed(1)

import logging
import lightgbm as lgb
import numpy as np
from pprint import pformat
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split

import bodmas.multiple_data as multiple_data
import bodmas.multiple_evaluate as evaluate
import bodmas.utils as utils
import bodmas.classifier as classifier
from bodmas.config import config
from bodmas.logger import init_log


SOPHOS_MODEL_FOLDER = config['sophos_model_folder']
SOPHOS_FEATURES_FOLDER = config['sophos_features_folder']


def main():
    # ----------------------------------------------- #
    # 0. Init log path and parse args                 #
    # ----------------------------------------------- #

    args = utils.parse_multiple_dataset_args()

    # the log file would be "./logs/multiple_main.log" and "./logs/multiple_main.log.wf" if no redirect
    log_path = './logs/multiple_main'
    if args.quiet:
        init_log(log_path, level=logging.INFO)
    else:
        init_log(log_path, level=logging.DEBUG)
    logging.warning('Running with configuration:\n' + pformat(vars(args)))
    logging.getLogger('matplotlib.font_manager').disabled = True

    task = args.task
    train_set = args.training_set
    test_begin_time, test_end_time = args.testing_time.split(',')
    SEED = args.seed
    families_cnt = args.families # useless, just ignore
    retrain = args.retrain
    clf = args.classifier

    # ----------------------------------------------- #
    # 1. global setting and folders                   #
    # ----------------------------------------------- #

    setting = args.setting_name
    GENERAL_DATA_FOLDER = f'multiple_data'
    DATA_FOLDER = f'multiple_data/{setting}'
    MODELS_FOLDER = f'multiple_models/{setting}'
    REPORT_FOLDER = f'multiple_reports/{setting}'
    os.makedirs(DATA_FOLDER, exist_ok=True)
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    os.makedirs(REPORT_FOLDER, exist_ok=True)

    # ----------------------------------------------- #
    # 2. Load the feature vectors and labels          #
    # ----------------------------------------------- #

    #### load testing set
    saved_data_path = os.path.join(GENERAL_DATA_FOLDER, 'bluehex_diversity_no',
                            f'X_and_y_test_{test_begin_time}_{test_end_time}_unnormalized.h5')
    X_train_origin, y_train_origin, \
        X_test_list, y_test_list = multiple_data.load_bluehex_data(task, test_begin_time, test_end_time,
                                                                   families_cnt=2, is_normalize=False,
                                                                   general_data_folder=GENERAL_DATA_FOLDER,
                                                                   setting_data_folder=None,
                                                                   saved_data_path=saved_data_path)

    #### training and validation set preparation
    model = None
    if train_set in ['ember', 'ucsb']:
        model_path = os.path.join(MODELS_FOLDER, f'{train_set}_{clf}_model_seed{SEED}.txt')
        if not os.path.exists(model_path):
            full_data_path = os.path.join(GENERAL_DATA_FOLDER, f'{train_set}.npz')
            t1 = timer()
            d = np.load(full_data_path)
            X, y = d['X'], d['y']
            t2 = timer()
            logging.debug(f'load {train_set} tik tok: {t2 - t1:.1f} seconds')
            logging.info(f'X: {X.shape}, y: {y.shape}')
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                              random_state=SEED, shuffle=True)
            logging.info(f'X_val: {X_val.shape}, y_val: {y_val.shape}')
            logging.info('validation and testing set prepared')
            if clf == 'gbdt':
                gbdt_clf = classifier.GBDTClassifier(saved_model_path=model_path)
                t1 = timer()
                model = gbdt_clf.train(X_train, y_train, task, families_cnt, retrain, config['gbdt_params'])
                t2 = timer()
                logging.debug(f'train {train_set} tik tok: {t2 - t1:.1f} seconds')
            else:
                raise ValueError(f'train_set {train_set} with classifier {clf} not implemented')
        else:
            model = lgb.Booster(model_file=model_path)
    elif train_set == 'sophos':
        if clf == 'gbdt':
            model_path = os.path.join(SOPHOS_MODEL_FOLDER, f'seed{SEED}', 'lightgbm.model')
            sophos_validatation_path = os.path.join(SOPHOS_FEATURES_FOLDER, 'validation-features.npz')
            t1 = timer()
            sophos_data = np.load(sophos_validatation_path)
            X_val = sophos_data['arr_0']
            y_val = sophos_data['arr_1']
            t2 = timer()
            logging.info(f'load sophos validation set: {t2 - t1:.1f} seconds')
            model = lgb.Booster(model_file=model_path)
        else:
            raise ValueError(f'train_set {train_set} with classifier {clf} not implemented')
    else:
        raise ValueError(f'train_set {train_set} not supported')

    # ----------------------------------------------- #
    # 3. Evaluate on validation and testing set       #
    # ----------------------------------------------- #
    ''' NOTE: use UCSB / Ember / Sophos validation set of their own '''

    fpr_list_all, tpr_list_all, f1_list_all = [], [], [] # each has two lists (fpr threshold 0.001 and 0.0001)

    t1 = timer()
    REPORT_PATH = os.path.join(REPORT_FOLDER, f'{train_set}_{clf}_seed{SEED}.csv')
    for fpr_target_on_val in [0.001, 0.0001]:
        threshold, fpr, tpr, f1 = evaluate.evaluate_prediction_on_validation(model, X_val, y_val,
                                                                             fpr_target_on_val,
                                                                             model_name=clf)

        fpr_list = [fpr]
        tpr_list = [tpr]
        f1_list = [f1]

        logging.critical(f'validation set threshold: {threshold}')

        for idx, X_test, y_test in zip(range(len(X_test_list)), X_test_list, y_test_list):
            phase = f'test_{idx}'
            ALL_CLASSIFICATION_RESULT_PATH = os.path.join(REPORT_FOLDER, 'intermediate',
                                                f'{train_set}_{clf}_test_{idx}_fpr{fpr_target_on_val}_all_classification_seed{SEED}.csv')
            utils.create_parent_folder(ALL_CLASSIFICATION_RESULT_PATH)
            MISCLASSIFIED_RESULT_PATH = os.path.join(REPORT_FOLDER, 'intermediate',
                                                f'{train_set}_{clf}_misclassified_test_{idx}_fpr{fpr_target_on_val}_seed{SEED}.csv')
            fpr, tpr, f1 = evaluate.evaluate_prediction_on_testing(model, phase, X_test, y_test, threshold,
                                                                    test_begin_time, test_end_time, SEED,
                                                                    ALL_CLASSIFICATION_RESULT_PATH,
                                                                    MISCLASSIFIED_RESULT_PATH,
                                                                    model_name=clf,
                                                                    detail=False)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            f1_list.append(f1)

        fpr_list_all.append(fpr_list)
        tpr_list_all.append(tpr_list)
        f1_list_all.append(f1_list)

    # add roc_auc_score for a fair comparison
    auc_score_list = evaluate.evaluate_auc_score(model, clf, X_val, y_val, X_test_list, y_test_list)
    evaluate.write_result_to_report(fpr_list_all, tpr_list_all, f1_list_all, auc_score_list, REPORT_PATH, is_rf=False)
    t2 = timer()
    logging.info(f'evaluate {train_set} tik tok: {t2 - t1:.1f} seconds')


if __name__ == "__main__":
    t1 = timer()
    main()
    t2 = timer()
    logging.info(f'total tik tok: {t2 - t1:.1f} seconds')
