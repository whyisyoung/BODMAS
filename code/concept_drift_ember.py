'''
/*
 * @Author: Limin Yang (liminy2@illinois.edu)
 * @Date: 2021-08-28 15:45:26
 * @Last Modified by: Limin Yang
 * @Last Modified time: 2021-08-28 22:01:06
 */

NOTE: script for Fig. 1.
    Transcend code was adapated from the original paper, please ask Feargus Pendlebury and Lorenzo Cavallaro for access.

    Accumulatively on 1 month explanation:
    Train on Ember, use random/transcend/probability to select r% from 19/10, test on 19/11
    Train on Ember, use random/transcend/probability to select r% from 19/11 + the previous selected samples from 19/10, test on 19/12
    Train on Ember, use random/transcend/probability to select r% from 19/12 + the previous selected samples from 19/10, 19/11, test on 20/01
    etc.
'''

import os

os.environ['PYTHONHASHSEED'] = '0'
from numpy.random import seed
import random
random.seed(1)
seed(1)


import sys
import logging

import lightgbm as lgb
import numpy as np
from collections import Counter
from pprint import pformat
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import bodmas.multiple_data as multiple_data
import bodmas.multiple_evaluate as evaluate
import bodmas.utils as utils
import bodmas.classifier as classifier
from bodmas.config import config
from bodmas.logger import init_log

# WARNING: you may comment the following 3 lines and corresponding code if you didn't ask access to the Transcend code
import transcend.detect as transcend_detect
sys.path.append('transcend')
import scores


def main():
    # ----------------------------------------------- #
    # 0. Init log path and parse args                 #
    # ----------------------------------------------- #

    args = utils.parse_drift_args()

    # the log file would be "./logs/concept_drift.log" and "./logs/concept_drift.log.wf" if no redirect
    log_path = './logs/concept_drift_ember'
    if args.quiet:
        init_log(log_path, level=logging.INFO)
    else:
        init_log(log_path, level=logging.DEBUG)
    logging.warning('Running with configuration:\n' + pformat(vars(args)))
    logging.getLogger('matplotlib.font_manager').disabled = True

    task = args.task
    test_begin_time, test_end_time = args.testing_time.split(',')
    families_cnt = args.families
    interval = args.month_interval

    sample_ratio = args.sample_ratio
    ember_ratio = args.ember_ratio  # NOTE: use 1.0 for proba and random, use 0.3 for Transcend
    SEED = args.seed # only for drift_random, pick up 1% samples using np.random.choice with this seed.

    # ----------------------------------------------- #
    # 1. global setting and folders                   #
    # ----------------------------------------------- #

    setting = args.setting_name  # use 'ember_drift_random', 'ember_drift_transcend', 'ember_drift_proba'
    GENERAL_DATA_FOLDER = f'multiple_data'
    DATA_FOLDER = f'multiple_data/{setting}'
    MODELS_FOLDER = f'multiple_models/{setting}'
    REPORT_FOLDER = f'multiple_reports/{setting}'
    os.makedirs(DATA_FOLDER, exist_ok=True)
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    os.makedirs(REPORT_FOLDER, exist_ok=True)

    MONTH_LIST = multiple_data.get_testing_month_list(test_begin_time, test_end_time)
    logging.info(f'MONTH_LIST: {MONTH_LIST}')

    # -------------------------------------------------------------------- #
    # 2. load ember dataset and the baseline model pre-trained on ember    #
    # -------------------------------------------------------------------- #

    # we will fix the follwing model as the base model (randomly picked), you may also change to multiple random modesl
    fixed_model_path = 'multiple_models/pretrain_ember/ember_gbdt_model_seed2.txt'
    fpr_threshold = 0.9922799999981189 # TODO: hard code for FPR 0.1%, taken from the validation set threshold in logs/pretrain_ember/seed_2_gbdt_xxx.log
    if not os.path.exists(fixed_model_path):
        logging.error(f'{fixed_model_path} not exist, need to execute "run_contro.sh" with ember as the training set first')
        sys.exit(-1)

    logging.info(f'load part/full of Ember data as initial training and validation set...')
    X_ember, y_ember = multiple_data.load_npz_data('ember', GENERAL_DATA_FOLDER)
    ember_total_num = X_ember.shape[0]
    if ember_ratio < 1.0:
        select_num = int(ember_total_num * ember_ratio)
        select_idx = np.random.choice(range(ember_total_num), size=select_num, replace=False)
        X_ember_part = X_ember[select_idx]
        y_ember_part = y_ember[select_idx]
        '''NOTE: use 24% Ember (30% * 80% = 24%) to train Transcend due to its high computation cost'''
        X_train_old, X_val, y_train_old, y_val = train_test_split(X_ember_part, y_ember_part, test_size=0.2, random_state=2, shuffle=True)
        logging.info(f'Ember selected: {select_num}, X_train_old: {X_train_old.shape}, X_val: {X_val.shape}, y_train_old: {y_train_old.shape}, y_val: {y_val.shape}')

    '''use 80% Ember as the base for accumulative retraining'''
    X_train_base, X_val_base, y_train_base, y_val_base = train_test_split(X_ember, y_ember, test_size=0.2, random_state=2, shuffle=True)

    logging.info(f'before adding {setting} sampling, training label: {Counter(y_train_base)}, validation label: {Counter(y_val_base)}')
    logging.info(f'load part of Ember data as initial training and validation set finished')


    # -------------------------------------------------------------------- #
    # 3. get the non-conformity measure score of the 24% selected ember    #
    # -------------------------------------------------------------------- #

    if 'transcend' in setting:
        # this could take 8 minutes, so it's not wise to put it in a 12 times loop.
        logging.info('Getting NCMs for train...')
        clf = lgb.Booster(model_file=fixed_model_path)
        train_ncms_path = os.path.join(REPORT_FOLDER, f'intermediate_ember_{ember_ratio}', 'training_ncms.p')
        if os.path.exists(train_ncms_path):
            train_ncms = transcend_detect.load_cached_data(train_ncms_path)
        else:
            train_ncms = scores.get_ncms(clf, X_train_old, y_train_old)
            transcend_detect.cache_data(train_ncms, train_ncms_path)
        logging.info(f'Getting NCMs for train done, type train_ncms: {type(train_ncms)}')

    # final feature vectors and labels for relabeling and testing set
    logging.info(f'start to load bluehex npz file...')
    X, y = multiple_data.load_npz_data('bluehex', GENERAL_DATA_FOLDER) # unnormalized data
    logging.info(f'start to load bluehex npz file finished')


    # ---------------------------------------------------------------------------------- #
    # 4. accumulatively sampling from the Blue Hexagon data and test on the next month   #
    # ---------------------------------------------------------------------------------- #

    if interval == 1:
        iteration = list(zip(range(1, len(MONTH_LIST)), MONTH_LIST[:-1], MONTH_LIST[1:]))
    else:
        sampling_list = evenly_split_list(MONTH_LIST[:-1], interval)
        test_list = evenly_split_list(MONTH_LIST[interval:], interval)
        iteration = list(zip(range(1, len(MONTH_LIST) // interval), sampling_list, test_list))

    for idx, sample_month, test_month in iteration:
        # ----------------------------------------------- #
        # 4.1. load data and define output folder         #
        # ----------------------------------------------- #

        logging.critical(f'{idx}, {sample_month}, {test_month}')
        if type(sample_month) is list:
            sample_month_str = '_'.join(sample_month)
            test_month_str = '_'.join(test_month)
        else:
            sample_month_str = sample_month
            test_month_str = test_month

        if 'random' in setting:
            POSTFIX = f'ember_{ember_ratio}_sample_{sample_month_str}_test_{test_month_str}_ratio_{sample_ratio}_random_{SEED}'
            MODEL_POSTFIX = f'_ember_{ember_ratio}_sample_{sample_month_str}_ratio_{sample_ratio}_random_{SEED}'
        else:
            POSTFIX = f'ember_{ember_ratio}_sample_{sample_month_str}_test_{test_month_str}_ratio_{sample_ratio}'
            MODEL_POSTFIX = f'_ember_{ember_ratio}_sample_{sample_month_str}_ratio_{sample_ratio}'

        SAVED_DATA_PATH = os.path.join(DATA_FOLDER, f'X_and_y_{POSTFIX}.h5')

        logging.info(f'idx-{idx} start to extract sampling month, test month data... ')

        X_sample_full, y_sample_full, X_test, y_test = \
            multiple_data.load_ember_drift_data(X, y, sample_month, test_month, GENERAL_DATA_FOLDER, SAVED_DATA_PATH)

        logging.info(f'idx-{idx} sampling, testing set prepared')

        # ----------------------------------------------- #
        # 4.2. sampleing from a month                     #
        # ----------------------------------------------- #

        sample_num = int(X_sample_full.shape[0] * sample_ratio)
        if 'random' in setting:
            seed(SEED)
            sample_idx = np.random.choice(range(X_sample_full.shape[0]), size=sample_num, replace=False)

            # record which samples were selected
            detect_path = os.path.join(REPORT_FOLDER, f'intermediate', f'{args.classifier}_detect_results_{sample_month_str}_random_{SEED}.csv')
            clf = lgb.Booster(model_file=fixed_model_path)
            y_sample_pred = clf.predict(X_sample_full)
            y_sample_prob = np.array([1 - t if t < 0.5 else t for t in y_sample_pred])
            y_sample_pred_label = np.array(y_sample_pred > fpr_threshold, dtype=np.int)

            with open(detect_path, 'w') as f:
                f.write(f'idx,real,pred,proba\n')
                for i in sample_idx:
                    f.write(f'{i},{y_sample_full[i]},{y_sample_pred_label[i]},{y_sample_prob[i]}\n')
        elif 'transcend' in setting:
            all_ood_idx = transcend_detect.get_ember_select_sample_idx(X_train_old, y_train_old,
                                                                        X_sample_full, y_sample_full,
                                                                        train_ncms,
                                                                        sample_month_str, args,
                                                                        test_begin_time,
                                                                        REPORT_FOLDER,
                                                                        fpr_threshold)
            sample_idx = all_ood_idx[:sample_num]
        elif 'probability' in setting:
            sample_idx_rank_by_proba = utils.get_ember_sample_idx_by_pred_proba(X_sample_full,
                                                                                y_sample_full,
                                                                                sample_month_str, args,
                                                                                test_begin_time,
                                                                                REPORT_FOLDER,
                                                                                fixed_model_path,
                                                                                fpr_threshold)
            sample_idx = sample_idx_rank_by_proba[:sample_num]
        else:
            raise f'invalid setting name {setting}'

        X_sample = X_sample_full[sample_idx]
        y_sample = y_sample_full[sample_idx]
        logging.info(f'idx-{idx} {setting} sampling: {X_sample.shape}, {y_sample.shape}')
        if idx == 1:
            X_sample_combine = np.copy(X_sample)
            y_sample_combine = np.copy(y_sample)
        else:
            X_sample_combine = np.vstack((X_sample_combine, X_sample))
            y_sample_combine = np.hstack((y_sample_combine, y_sample))

        logging.critical(f'idx-{idx} accumulated sampling cnt: {X_sample_combine.shape}, {y_sample_combine.shape}')

        '''use 80% Ember as the base for accumulative retraining'''
        X_train = np.vstack((X_train_base, X_sample_combine))
        y_train = np.hstack((y_train_base, y_sample_combine))


        logging.info(f'idx-{idx} after adding sampling, training: {X_train.shape}, validation: {X_val_base.shape}')
        logging.info(f'idx-{idx} training label: {Counter(y_train)}')
        logging.info(f'idx-{idx} validation label: {Counter(y_val_base)}')
        logging.info(f'idx-{idx} testing set label: {Counter(y_test)}')

        # ----------------------------------------------- #
        # 4.3. Train the classifier                         #
        # ----------------------------------------------- #

        # lgbm_model = classifier.train_model(X_train, y_train, SAVED_MODEL_PATH, config['gbdt_params'], redo_flag=False)

        clf = args.classifier
        retrain = args.retrain

        if clf == 'gbdt':
            SAVED_MODEL_PATH = os.path.join(MODELS_FOLDER, f'gbdt{MODEL_POSTFIX}.txt')
            gbdt_clf = classifier.GBDTClassifier(saved_model_path=SAVED_MODEL_PATH)
            logging.debug(f'model path: {SAVED_MODEL_PATH}')
            model = gbdt_clf.train(X_train, y_train, task, families_cnt, retrain, config['gbdt_params'])
        elif clf == 'rf':
            tree = args.tree
            SAVED_MODEL_PATH = os.path.join(MODELS_FOLDER, f'rf_tree{tree}{MODEL_POSTFIX}.pkl')
            rf_clf = classifier.RFClassifier(SAVED_MODEL_PATH, tree)
            model = rf_clf.train(X_train, y_train, retrain)
        else:
            raise ValueError(f'classifier {clf} not supported')

        # ----------------------------------------------- #
        # 4.4. Evaluate the classifier                      #
        # ----------------------------------------------- #

        pred_begin = timer()
        REPORT_PATH = os.path.join(REPORT_FOLDER, f'ratio_{sample_ratio}', f'{clf}_{task}_report_{POSTFIX}.csv')
        utils.create_parent_folder(REPORT_PATH)

        if clf != 'rf':
            fpr_list_all, tpr_list_all, f1_list_all = [], [], [] # each has two lists (fpr threshold 0.01 and 0.001)

            for fpr_target_on_val in [0.001, 0.0001]:
                threshold, fpr, tpr, f1 = evaluate.evaluate_prediction_on_validation(model, X_val_base, y_val_base,
                                                                                        fpr_target_on_val,
                                                                                        model_name=clf)

                fpr_list = [fpr]
                tpr_list = [tpr]
                f1_list = [f1]

                logging.critical(f'idx-{idx} validation set threshold: {threshold}')

                phase = test_month
                ALL_CLASSIFICATION_RESULT_PATH = os.path.join(REPORT_FOLDER, 'intermediate',
                                                                f'{clf}_test_fpr{fpr_target_on_val}_all_classification_result_{POSTFIX}.csv')
                utils.create_parent_folder(ALL_CLASSIFICATION_RESULT_PATH)
                MISCLASSIFIED_RESULT_PATH = os.path.join(REPORT_FOLDER, 'intermediate',
                                                                f'misclassified_{clf}_test_fpr{fpr_target_on_val}_result_{POSTFIX}.csv')
                fpr, tpr, f1 = evaluate.evaluate_prediction_on_testing(model, phase, X_test, y_test, threshold,
                                                                        test_begin_time, test_end_time, SEED,
                                                                        ALL_CLASSIFICATION_RESULT_PATH,
                                                                        MISCLASSIFIED_RESULT_PATH,
                                                                        model_name=clf, detail=False)
                fpr_list.append(fpr)
                tpr_list.append(tpr)
                f1_list.append(f1)

                fpr_list_all.append(fpr_list)
                tpr_list_all.append(tpr_list)
                f1_list_all.append(f1_list)

            # add roc_auc_score for a fair comparison
            auc_score_list = evaluate.evaluate_auc_score(model, clf, X_val_base, y_val_base, [X_test], [y_test])
            evaluate.write_result_to_report(fpr_list_all, tpr_list_all, f1_list_all, auc_score_list, REPORT_PATH, is_rf=False)
        else:
            '''do not use FPR threshold for Random Forest classifier because it's a majority vote, threshold seems meaningless.'''
            evaluate.evaluate_rf_model_performance(model, X_val_base, y_val_base, [X_test], [y_test], REPORT_PATH)

        pred_end = timer()
        logging.info(f'prediction on validation and testing time: {pred_end - pred_begin:.1f} seconds')


def evenly_split_list(list1, interval):
    # remove the last element if cannot evenly split
    list2 = [list1[i:i+interval] for i in range(0, len(list1), interval)]
    if len(list2[-1]) != interval:
        list2 = list2[:-1]
    return list2


if __name__ == "__main__":
    start = timer()
    main()
    end = timer()
    logging.info(f'time elapsed: {end - start:.2f}')
