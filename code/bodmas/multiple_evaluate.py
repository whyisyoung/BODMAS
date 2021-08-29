import os, sys
import logging
import traceback

from timeit import default_timer as timer
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

import bodmas.multiple_data as multiple_data
import bodmas.utils as utils


def get_fpr(y_true, y_pred):
    nbenign = (np.array(y_true) == 0).sum()
    nfalse = (y_pred[y_true == 0] == 1).sum()
    return nfalse / float(nbenign)


def find_threshold(y_true, y_pred, fpr_target):
    thresh = 0.0
    fpr = get_fpr(y_true, y_pred > thresh)
    cnt = 0
    while fpr > fpr_target and thresh < 1.0:
        thresh += 0.00001
        fpr = get_fpr(y_true, y_pred > thresh)
        cnt += 1
        if cnt % 10000 == 0:
            logging.debug(f'still running... thresh: {thresh}')
    return thresh, fpr


def evaluate_prediction_on_validation(model, X_val, y_val, fpr_target_on_val, model_name):
    '''
    NOTE:
        do not use FPR threshold for Random Forest classifier because it's a majority vote, threshold seems meaningless.
    '''
    y_val_pred = model.predict(X_val)
    logging.debug(f'y_val_pred.shape: {y_val_pred.shape}')
    if model_name == 'mlp':
        y_val_pred = np.array([float(v[0]) for v in y_val_pred])

    t1 = timer()
    threshold, fpr = find_threshold(y_val, y_val_pred, fpr_target_on_val)
    t2 = timer()
    logging.info(f'find threshold on validation set: {t2 - t1:.1f} seconds')

    cm = confusion_matrix(y_val, y_val_pred > threshold)
    TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    tpr = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1 = f1_score(y_val, y_val_pred > threshold)
    logging.info(f'validation fpr: {fpr}, tpr: {tpr}, confusion matrix:\n {cm}')

    ''' NOTE: the threshold got on validation set should directly applied to the testing set'''
    return threshold, fpr, tpr, f1


def evaluate_prediction_on_testing(model, phase, X_test, y_test, threshold, test_begin_time, test_end_time, seed,
                                   all_classification_result_path, misclassified_result_path, model_name, detail=True):

    '''
    NOTE:
        do not use FPR threshold for Random Forest classifier because it's a majority vote, threshold seems meaningless.
    '''
    if model_name == 'rf':
        raise ValueError('Random Forest should not use the FPR threshold from valiation set')

    y_test_pred = model.predict(X_test)
    if model_name == 'mlp':
        y_test_pred = np.array([float(v[0]) for v in y_test_pred])

    # use the threshold determined by the validation set
    cm = confusion_matrix(y_test, y_test_pred > threshold)
    TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    tpr = TP / (TP + FN)
    fpr = FP / (FP + TN)
    f1 = f1_score(y_test, y_test_pred > threshold)
    logging.critical(f'{phase} confusion matrix: \n {cm}')

    if detail:
        report_detailed_classification_result(y_test, y_test_pred, threshold, phase, test_begin_time, test_end_time, seed,
                                              misclassified_result_path, all_classification_result_path)

    return fpr, tpr, f1


def evaluate_auc_score(model, clf_name, X_val, y_val, X_test_list, y_test_list):
    auc_score_list = [calc_roc_score(clf_name, model, X_val, y_val)]

    for idx, X_test, y_test in zip(range(len(X_test_list)), X_test_list, y_test_list):
        auc_score = calc_roc_score(clf_name, model, X_test, y_test)
        auc_score_list.append(auc_score)

    return auc_score_list


def calc_roc_score(clf_name, model, X, y_true):
    if clf_name == 'gbdt':
        y_score = model.predict(X)
    elif clf_name == 'rf':
        # The binary case expects a shape (n_samples,), and the scores must be the scores of the class with the greater label.
        y_score = model.predict_proba(X)[:, 1]
    else:
        raise f'classifier {clf_name} y_score not implemented'
    auc_score = roc_auc_score(y_true, y_score)
    return auc_score


def write_result_to_report(fpr_list, tpr_list, f1_list, auc_score_list, report_path, is_rf=False):
    phase_list = ['val'] + [f'test_{i}' for i in range(len(auc_score_list) - 1)]
    with open(report_path, 'w') as f:
        if is_rf:
            f.write('phase,fpr,tpr,f1,auc_score\n')
            for i in range(len(fpr_list)):
                phase = phase_list[i]
                fpr = fpr_list[i]
                tpr = tpr_list[i]
                f1 = f1_list[i]
                auc_score = auc_score_list[i]
                f.write(f'{phase},{fpr*100:.4f}%,{tpr*100:.2f}%,{f1*100:.2f}%,{auc_score*100:.4f}%\n')
        else:
            f.write('phase,fpr_0.1%,tpr_0.1%,f1_0.1%,fpr_0.01%,tpr_0.01%,f1_0.01%,auc_score\n')
            for i in range(len(phase_list)):
                phase = phase_list[i]
                fpr_1 = fpr_list[0][i]
                fpr_2 = fpr_list[1][i]
                tpr_1 = tpr_list[0][i]
                tpr_2 = tpr_list[1][i]
                f1_1 = f1_list[0][i]
                f1_2 = f1_list[1][i]
                auc_score = auc_score_list[i]
                f.write(f'{phase},{fpr_1*100:.4f}%,{tpr_1*100:.2f}%,{f1_1*100:.2f}%,' + \
                        f'{fpr_2*100:.4f}%,{tpr_2*100:.2f}%,{f1_2*100:.2f}%,{auc_score*100:.4f}%\n')
    logging.info(f'write result to {report_path} done')


def evaluate_rf_model_performance(model, X_val, y_val, X_test_list, y_test_list, report_path):
    y_val_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_val_pred)
    TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    tpr = TP / (TP + FN)
    fpr = FP / (FP + TN)
    f1 = f1_score(y_val, y_val_pred)
    auc_score = calc_roc_score('rf', model, X_val, y_val)
    fpr_list, tpr_list, f1_list, auc_score_list = [fpr], [tpr], [f1], [auc_score]

    for idx, X_test, y_test in zip(range(len(X_test_list)), X_test_list, y_test_list):
        y_test_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_test_pred)
        TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        tpr = TP / (TP + FN)
        fpr = FP / (FP + TN)
        f1 = f1_score(y_test, y_test_pred)
        auc_score = calc_roc_score('rf', model, X_test, y_test)

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        f1_list.append(f1)
        auc_score_list.append(auc_score)

    write_result_to_report(fpr_list, tpr_list, f1_list, auc_score_list, report_path, is_rf=True)


def report_detailed_classification_result(y_test, y_test_pred, threshold, phase, test_begin_time, test_end_time, seed,
                                          misclassified_result_path, all_classification_result_path):
    MONTH_LIST = multiple_data.get_testing_month_list(test_begin_time, test_end_time)
    with open(misclassified_result_path, 'w') as f:
        f.write('sample_idx,sha,real_label,pred_label,pred_prob,timestamp,family,family_seen_type\n')
        meta_df = multiple_data.load_meta('bluehex', general_data_folder='multiple_data')
        test_begin_time = MONTH_LIST[0]
        meta_train_origin = meta_df[meta_df.timestamp.str[:7] < test_begin_time]

        month_idx = int(phase.replace('test_', ''))
        month = MONTH_LIST[month_idx]
        meta_test = meta_df[meta_df.timestamp.str[:7] == month]

        sha_train_origin, ts_train_origin, family_train_origin = get_sha_ts_family_list(meta_train_origin)
        sha_train, sha_val, family_train, family_val = train_test_split(sha_train_origin, family_train_origin,
                                                                        test_size=0.2, random_state=seed, shuffle=True)
        sha_test, ts_test, family_test = get_sha_ts_family_list(meta_test)

        logging.debug(f'month: {month}')
        logging.debug(f'y_test: {y_test.shape}')
        logging.debug(f'sha_test: {sha_test.shape}')

        FN = 0
        FP = 0
        for idx in range(len(y_test)):
            y_true = y_test[idx]
            y_pred = 1 if y_test_pred[idx] > threshold else 0
            if y_true != y_pred:
                sha = sha_test[idx]
                ts = ts_test[idx]
                family = family_test[idx]
                if y_true == 0:
                    family = 'benign'
                    family_type = ''
                    FP += 1
                else:
                    FN += 1
                    if family not in family_train:
                        family_type = 'New'
                    else:
                        family_type = 'SeenBefore'
                f.write(f'{idx},{sha},{y_true},{y_pred},{y_test_pred[idx]:.6f},{ts},{family},{family_type}\n')

    df = pd.read_csv(misclassified_result_path, header=0)
    fp_df = df[(df.real_label == 0) & (df.pred_label == 1)]
    fn_df = df[(df.real_label == 1) & (df.pred_label == 0)]

    report_misclassified_helper(fp_df, FP, 'fp', misclassified_result_path)
    report_misclassified_helper(fn_df, FN, 'fn', misclassified_result_path)

    # report the full classificationr results.
    with open(all_classification_result_path, 'w') as f:
        f.write('sample_idx,sha,real_label,pred_label,pred_prob,family,type\n') # pred_prob is the direct output of GBDT, not the real probability
        for idx in range(len(y_test)):
            y_true = y_test[idx]
            y_pred = 1 if y_test_pred[idx] > threshold else 0
            sha = sha_test[idx]
            family = family_test[idx]
            if y_true == 0:
                family_type = ''
            else:
                if family not in family_train:
                    family_type = 'New'
                else:
                    family_type = 'SeenBefore'
            f.write(f'{idx},{sha},{y_true},{y_pred},{y_test_pred[idx]:.6f},{family},{family_type}\n')


def report_misclassified_helper(df, count, error_type, misclassified_result_path):
    error_summary = [(family, family_type) for family, family_type in zip(df.family, df.family_seen_type)]
    with open(misclassified_result_path, 'a') as f:
        f.write('=' * 40 + '\n')
        if error_type == 'fp':
            f.write(f'False Positive ({count}):\n')
        else:
            f.write(f'False Negative ({count}):\n')
            family_seen_type = df.family_seen_type.to_numpy()
            f.write(f'{Counter(family_seen_type)}\n\n')
        c = Counter(error_summary).most_common()
        for k, v in c:
            f.write(f'{k}\t{v}\n')


def get_sha_ts_family_list(meta_df):
    sha_list = meta_df['sha'].to_numpy()
    ts_list = meta_df['timestamp'].to_numpy()
    family_list = meta_df['family'].to_numpy()
    return sha_list, ts_list, family_list



def evaluate_multiclass_prediction(y, y_pred, unseen_family, top_n_acc, cm_fig_path, mapping, phase):
    '''roc_auc_score is not a good metric for multi-class classification because you need to calculate roc_auc_score
    for each label. Instead, confusion matrix is a better metric.'''
    y_pred = replace_as_correct_or_not(y, y_pred)
    logging.debug(f'evaluate_multiclass_prediction y shape: {y.shape}, y_pred shape: {y_pred.shape}')
    acc = accuracy_score(y, y_pred)
    if phase == 'val':
        cm = confusion_matrix(y, y_pred, labels=range(unseen_family))
    else:
        cm = confusion_matrix(y, y_pred, labels=range(unseen_family + 1))

    inclass_family_idx = np.where(y != unseen_family)[0]
    y_inclass = y[inclass_family_idx]
    y_inclass_pred = y_pred[inclass_family_idx]

    inclass_acc = accuracy_score(y_inclass, y_inclass_pred)
    inclass_cm = confusion_matrix(y_inclass, y_inclass_pred)

    logging.critical(f'{phase} topacc: {top_n_acc} acc: \n {acc:.4f}')
    logging.critical(f'{phase} topacc: {top_n_acc} confusion matrix: \n {cm}')
    logging.critical(f'{phase} topacc: {top_n_acc} inclass family acc: \n {inclass_acc:.4f}')
    logging.critical(f'{phase} topacc: {top_n_acc} inclass family confusion matrix: \n {inclass_cm}')

    # write_multiclass_result_to_report(acc, cm, inclass_acc, inclass_cm, report_path, phase, top_n_acc)

    plot_confusion_matrix(cm, unseen_family, mapping, cm_fig_path, phase)
    return acc, inclass_acc


def replace_as_correct_or_not(y, y_pred):
    y_pred_prime = np.copy(y)
    for i in range(y.shape[0]):
        if y[i] not in y_pred[i]:
            y_pred_prime[i] = y_pred[i][0]
    return y_pred_prime


def write_multiclass_result_to_report(acc_final_list, inclass_acc_final_list, report_path):
    phase_list = ['val'] + [f'test_{i}' for i in range(acc_final_list.shape[0] - 1)]
    with open(report_path, 'w') as f:
        f.write('phase')
        for i in range(1, acc_final_list.shape[1]+1):
            f.write(f',topacc_{i}')
        for i in range(1, inclass_acc_final_list.shape[1]+1):
            f.write(f',inclass_topacc_{i}')
        f.write('\n')

        for i, phase in enumerate(phase_list):
            f.write(f'{phase}')
            for j in range(acc_final_list.shape[1]):
                f.write(f',{acc_final_list[i][j]:.4f}')
            for j in range(acc_final_list.shape[1]):
                f.write(f',{inclass_acc_final_list[i][j]:.4f}')
            f.write('\n')


def plot_confusion_matrix(cm, top_family, mapping, save_fig_name, phase):
    logging.getLogger('matplotlib.font_manager').disabled = True
    fig, ax = plt.subplots(figsize=(25,25))
    fontsize = 12
    ax = sns.heatmap(cm, annot=True, fmt="d", annot_kws={"size": fontsize})

    if phase == 'val':
        no_of_axes = top_family
    else:
        no_of_axes = top_family + 1
    logging.debug(f'no_of_axes: {no_of_axes}')

    ax.set_xticks(np.arange(no_of_axes) + 0.5)
    ax.set_yticks(np.arange(no_of_axes) + 0.5)
    if phase == 'val':
        ax.set_xticklabels([mapping[str(t)] for t in range(no_of_axes)], fontsize=fontsize, rotation='vertical')
        ax.set_yticklabels([mapping[str(t)] for t in range(no_of_axes)], fontsize=fontsize, rotation='horizontal')
    else:
        ax.set_xticklabels([mapping[str(t)] for t in range(no_of_axes-1)] + ['unseen'], fontsize=fontsize, rotation='vertical')
        ax.set_yticklabels([mapping[str(t)] for t in range(no_of_axes-1)] + ['unseen'], fontsize=fontsize, rotation='horizontal')

    ax.set_title("Confusion matrix", fontsize=20)
    ax.set_xlabel('Predicted label', fontsize=16)
    ax.set_ylabel('True label', fontsize=16)

    fig.tight_layout()
    utils.create_parent_folder(save_fig_name)
    fig.savefig(save_fig_name, dpi=200)
    plt.clf()
