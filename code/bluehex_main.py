'''
/*
 * @Author: Limin Yang (liminy2@illinois.edu)
 * @Date: 2021-08-29 00:44:22
 * @Last Modified by: Limin Yang
 * @Last Modified time: 2021-08-29 03:49:06
 */

NOTE: get data for Fig. 2, 3, and 4.
    * ember-2018 does not provide reliable family labels, so consider it has no family info.

'''

import os

os.environ['PYTHONHASHSEED'] = '0'
from numpy.random import seed
import random
random.seed(1)
seed(1)

import sys
import logging
import traceback

import numpy as np
import matplotlib.pylab as plt
from collections import Counter
from pprint import pformat
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split

import bodmas.multiple_data as multiple_data
import bodmas.multiple_evaluate as evaluate
import bodmas.utils as utils
import bodmas.classifier as classifier
from bodmas.config import config
from bodmas.logger import init_log


MAX_TOP_N_ACC = 5


def main():
    # ----------------------------------------------- #
    # 0. Init log path and parse args                 #
    # ----------------------------------------------- #

    args = utils.parse_multiple_dataset_args()

    # the log file would be "./logs/bluehex_main.log" and "./logs/bluehex_main.log.wf" if no redirect
    log_path = './logs/bluehex_main'
    if args.quiet:
        init_log(log_path, level=logging.INFO)
    else:
        init_log(log_path, level=logging.DEBUG)
    logging.warning('Running with configuration:\n' + pformat(vars(args)))
    logging.getLogger('matplotlib.font_manager').disabled = True

    # ----------------------------------------------- #
    # 1. global setting and load data                 #
    # ----------------------------------------------- #
    setting = args.setting_name
    GENERAL_DATA_FOLDER = f'multiple_data'
    DATA_FOLDER = f'multiple_data/{setting}'
    MODELS_FOLDER = f'multiple_models/{setting}'
    FIG_FOLDER = f'multiple_fig/{setting}'
    REPORT_FOLDER = f'multiple_reports/{setting}'
    os.makedirs(DATA_FOLDER, exist_ok=True)
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    os.makedirs(FIG_FOLDER, exist_ok=True)
    os.makedirs(REPORT_FOLDER, exist_ok=True)

    task = args.task
    train_set = args.training_set
    diversity = args.diversity # only set value as "no" for this argument
    test_begin_time, test_end_time = args.testing_time.split(',')
    families_cnt = args.families
    SEED = args.seed

    POSTFIX, MODEL_POSTFIX = get_saved_file_postfix(task, diversity, test_begin_time, test_end_time,
                                                    train_set, families_cnt)

    # final feature vectors and labels for training and testing set
    SAVED_DATA_PATH = os.path.join(DATA_FOLDER, f'X_and_y_{POSTFIX}_r{SEED}.h5')


    X_train_origin, y_train_origin, X_test_list, y_test_list = \
        multiple_data.load_bluehex_data(task, test_begin_time, test_end_time, families_cnt,
                                        is_normalize=True,
                                        general_data_folder=GENERAL_DATA_FOLDER,
                                        setting_data_folder=DATA_FOLDER,
                                        saved_data_path=SAVED_DATA_PATH)


    logging.info('training and testing set prepared')
    X_train, X_val, y_train, y_val = train_test_split(X_train_origin, y_train_origin, test_size=0.2,
                                                      random_state=SEED, shuffle=True)

    NUM_FEATURES = X_train.shape[1]

    logging.info(f'after split, training: {X_train.shape}, validation: {X_val.shape}')
    logging.info(f'training label: {Counter(y_train)}')
    logging.info(f'validation label: {Counter(y_val)}')
    for idx, y_test in enumerate(y_test_list):
        logging.info(f'testing set {idx} label: {Counter(y_test)}')

    # ----------------------------------------------- #
    # 2. Train the classifier                         #
    # ----------------------------------------------- #

    clf = args.classifier
    retrain = args.retrain

    if clf == 'gbdt':
        SAVED_MODEL_PATH = os.path.join(MODELS_FOLDER, f'gbdt{MODEL_POSTFIX}_r{SEED}.txt')
        gbdt_clf = classifier.GBDTClassifier(saved_model_path=SAVED_MODEL_PATH)
        model = gbdt_clf.train(X_train, y_train, task, families_cnt, retrain, config['gbdt_params'])
    elif clf == 'mlp':
        # NOTE: multi-class MLP not implemented
        dims = utils.get_model_dims('MLP', NUM_FEATURES, args.mlp_hidden, 1)
        dims_str = '-'.join(map(str, dims))
        lr = args.mlp_lr
        batch = args.mlp_batch_size
        epochs = args.mlp_epochs
        dropout = args.mlp_dropout
        SAVED_MODEL_PATH = os.path.join(MODELS_FOLDER, f'mlp_{dims_str}_lr{lr}_b{batch}_e{epochs}_d{dropout}{MODEL_POSTFIX}.h5')
        mlp_clf = classifier.MLPClassifier(SAVED_MODEL_PATH, dims, dropout, verbose=1)
        model = mlp_clf.train(X_train, y_train, X_val, y_val, retrain, lr, batch, epochs)
    else: # clf == 'rf':
        tree = args.tree
        SAVED_MODEL_PATH = os.path.join(MODELS_FOLDER, f'rf_tree{tree}{MODEL_POSTFIX}.pkl')
        rf_clf = classifier.RFClassifier(SAVED_MODEL_PATH, tree)
        model = rf_clf.train(X_train, y_train, retrain)

    # ----------------------------------------------- #
    # 3. Evaluate the classifier                      #
    # ----------------------------------------------- #

    pred_begin = timer()
    REPORT_PATH = os.path.join(REPORT_FOLDER, f'{clf}_{task}_report_{POSTFIX}_r{SEED}.csv')

    if task == 'binary':
        if clf != 'rf':
            fpr_list_all, tpr_list_all, f1_list_all = [], [], [] # each has two lists (fpr threshold 0.01 and 0.001)
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
                                                                f'{clf}_test_{idx}_fpr{fpr_target_on_val}_all_classification_result_{POSTFIX}.csv')
                    utils.create_parent_folder(ALL_CLASSIFICATION_RESULT_PATH)
                    MISCLASSIFIED_RESULT_PATH = os.path.join(REPORT_FOLDER, 'intermediate',
                                                                  f'misclassified_{clf}_test_{idx}_fpr{fpr_target_on_val}_result.csv')
                    fpr, tpr, f1 = evaluate.evaluate_prediction_on_testing(model, phase, X_test, y_test, threshold,
                                                                           test_begin_time, test_end_time, SEED,
                                                                           ALL_CLASSIFICATION_RESULT_PATH,
                                                                           MISCLASSIFIED_RESULT_PATH,
                                                                           model_name=clf)
                    fpr_list.append(fpr)
                    tpr_list.append(tpr)
                    f1_list.append(f1)

                fpr_list_all.append(fpr_list)
                tpr_list_all.append(tpr_list)
                f1_list_all.append(f1_list)

            # add roc_auc_score for a fair comparison
            auc_score_list = evaluate.evaluate_auc_score(model, clf, X_val, y_val, X_test_list, y_test_list)
            evaluate.write_result_to_report(fpr_list_all, tpr_list_all, f1_list_all, auc_score_list, REPORT_PATH, is_rf=False)
        else:
            '''do not use FPR threshold for Random Forest classifier because it's a majority vote, threshold seems meaningless.'''
            evaluate.evaluate_rf_model_performance(model, X_val, y_val, X_test_list, y_test_list, REPORT_PATH)
    else:
        '''multi-class classification'''
        label_mapping_file = os.path.join(DATA_FOLDER, f'top_{families_cnt}_label_mapping.json')
        mapping = utils.load_json(label_mapping_file)

        acc_final_list = []
        inclass_acc_final_list = []
        for top_n_acc in range(1, MAX_TOP_N_ACC+1):
            acc_list = []
            inclass_acc_list = []
            '''top 2 acc means if one of two predicted labels with the biggest probabilities match
            with the ground-truth label, then we consider the prediction as correct.'''

            '''validation set'''
            acc, inclass_acc = multiclass_prediction_helper(clf, model, X_val, y_val, families_cnt,
                                                            top_n_acc, mapping, REPORT_FOLDER, phase='val')
            logging.critical(f'top_n_acc: {top_n_acc}, validation: acc {acc:.4f}, inclass_acc: {inclass_acc:.4f}')
            acc_list.append(acc)
            inclass_acc_list.append(inclass_acc)

            '''testing sets'''
            for idx, X_test, y_test in zip(range(len(X_test_list)), X_test_list, y_test_list):
                acc, inclass_acc = multiclass_prediction_helper(clf, model, X_test, y_test, families_cnt,
                                                                top_n_acc, mapping, REPORT_FOLDER, phase=f'test_{idx}')
                logging.critical(f'top_n_acc: {top_n_acc}, test-{idx}: acc {acc:.4f}, inclass_acc: {inclass_acc:.4f}')
                acc_list.append(acc)
                inclass_acc_list.append(inclass_acc)

            acc_final_list.append(acc_list)
            inclass_acc_final_list.append(inclass_acc_list)

        acc_final_list = np.transpose(np.array(acc_final_list)) # row: phase, column: topacc and inclass_acc
        inclass_acc_final_list = np.transpose(np.array(inclass_acc_final_list))
        evaluate.write_multiclass_result_to_report(acc_final_list, inclass_acc_final_list, REPORT_PATH)

    pred_end = timer()
    logging.info(f'prediction on validation and testing time: {pred_end - pred_begin:.1f} seconds')


def multiclass_prediction_helper(clf, model, X, y, families_cnt, top_n_acc, mapping, report_folder, phase):
    if clf == 'rf':
        # RF supports multi-class classification internally, so need to use predict_proba() to get the probabilities
        y_pred = model.predict_proba(X)
    else:
        y_pred = model.predict(X)

    unseen_family = families_cnt

    ''' use -1 to sort in descending order.
        another solution is to use np.flip(y_pred, axis=1)
        y_pred[::-1] would reverse with axis=0
    '''
    y_pred = np.argsort(-1 * y_pred, axis=1)[:, :top_n_acc]
    logging.debug(f'y_{phase}_pred shape: {y_pred.shape}')

    FIG_SAVE_FOLDER = os.path.join(report_folder, 'intermediate')
    utils.create_folder(FIG_SAVE_FOLDER)
    VAL_CM_FIG_PATH = os.path.join(FIG_SAVE_FOLDER, f'{clf}_family_{families_cnt}_topacc_{top_n_acc}_{phase}_confusion_matrix.png')

    acc, inclass_acc = evaluate.evaluate_multiclass_prediction(y, y_pred, unseen_family, top_n_acc,
                                                                VAL_CM_FIG_PATH, mapping, phase=phase)

    return acc, inclass_acc


def get_saved_file_postfix(task, diversity, test_begin_time, test_end_time, train_set, families_cnt):
    if task == 'binary':
        if diversity == 'no':
            POSTFIX = f'test_{test_begin_time}_{test_end_time}'
            MODEL_POSTFIX = ''
    else:
        POSTFIX = f'{train_set}_families_{families_cnt}_test_{test_begin_time}_{test_end_time}'
        MODEL_POSTFIX = f'_{train_set}_families_{families_cnt}'

    return POSTFIX, MODEL_POSTFIX


if __name__ == "__main__":
    start = timer()
    main()
    end = timer()
    logging.info(f'time elapsed: {end - start:.2f}')
