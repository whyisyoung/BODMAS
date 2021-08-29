"""
multiple_data.py
~~~~~~~

Functions for cleaning, loading, and caching data from multiple datasets.

"""
import os
os.environ['PYTHONHASHSEED'] = '0'
from numpy.random import seed
import random
random.seed(1)
seed(1)

import sys
import logging


from collections import Counter
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import psutil
import h5py
# import ember
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import bodmas.utils as utils


def load_bluehex_data(task, test_begin_time, test_end_time, families_cnt, is_normalize,
                      general_data_folder, setting_data_folder, saved_data_path):
    if os.path.exists(saved_data_path):
        t1 = timer()
        with h5py.File(saved_data_path, 'r') as hf:
            X_train = np.array(hf.get('X_train'))
            y_train = np.array(hf.get('y_train'))
            test = hf.get('test')
            X_test_list, y_test_list = [], []
            for i in range(len(list(test.keys())) // 2):
                X_test_list.append(np.array(test.get(f'X_test_{i}')))
                y_test_list.append(np.array(test.get(f'y_test_{i}')))
        t2 = timer()
        logging.debug(f'load h5py time: {t2 - t1:.1f} seconds')
    else:
        begin = timer()

        if task == 'binary':
            ''' define which samples should be used from the first 3 months bluehex training set (could be all or part of it) '''
            meta_df = load_meta('bluehex', general_data_folder)
            meta_train = meta_df[meta_df.timestamp.str[:7] < test_begin_time]
            logging.debug(f'meta_train: {meta_train.shape}')

            logging.info(f'loading bluehex training data ...')
            X, y = load_npz_data('bluehex', general_data_folder)
            X_train = np.array(X[meta_train.index], dtype=np.float32)
            y_train = np.array(y[meta_train.index], dtype=np.float32)

            logging.info(f'loading bluehex training data finished ...')

            ''' prepare a list of testing sets from bluehex (each month as a testing set), fixed for all settings '''
            X_test_list, y_test_list = prepare_testing_set(test_begin_time, test_end_time, general_data_folder)

            # NOTE: Normalization is only needed for MLP classifier,
            # it's NOT needed for GBDT and Random Forest (if normalized, it will NOT affect the results)
            if is_normalize:
                X_train, X_test_list = normalize_training_testing(X_train, X_test_list)
        else:
            ''' multi-class classification preparing training and testing sets'''
            meta_df = load_meta('bluehex', general_data_folder)
            meta_train = get_family_training_meta(meta_df, test_begin_time, families_cnt, task='multiclass')
            X, _ = load_npz_data('bluehex', general_data_folder)
            X_train = X[meta_train.index]
            y_train_raw = meta_train.family.to_numpy()
            logging.debug(f'before label encoding: {Counter(y_train_raw)}')

            '''transform training set to continuous labels'''
            le = LabelEncoder()
            y_train = le.fit_transform(y_train_raw)
            mapping = {}
            inv_mapping = {}
            for i in range(len(y_train)):
                mapping[y_train_raw[i]] = y_train[i]  # mapping: real label -> converted label
                inv_mapping[str(y_train[i])] = y_train_raw[i] # inv_mapping: converted label -> real label
            logging.debug(f'LabelEncoder mapping: {mapping}')
            logging.debug(f'after relabeling training: {Counter(y_train)}')
            utils.dump_json(inv_mapping, setting_data_folder, f'top_{families_cnt}_label_mapping.json')

            PERSISTENT_NEW_FAMILY = families_cnt
            y_test_list = []
            X_test_list, y_test_raw_list = prepare_testing_set(test_begin_time, test_end_time, general_data_folder, task='multiclass')
            for idx, y_test_raw in enumerate(y_test_raw_list):
                y_test = np.zeros(shape=y_test_raw.shape, dtype=np.int32)
                for i in range(len(y_test_raw)):
                    if y_test_raw[i] not in y_train_raw:  # new family
                        y_test[i] = PERSISTENT_NEW_FAMILY
                    else:
                        y_test[i] = mapping[y_test_raw[i]]
                logging.debug(f'test-{idx} after relabeling testing: {Counter(y_test)}')
                y_test_list.append(y_test)

            y_train = np.array(y_train, dtype=np.int32)

            # NOTE: Normalization is only needed for MLP classifier,
            # it's NOT needed for GBDT and Random Forest (if normalized, it will NOT affect the results)
            if is_normalize:
                X_train, X_test_list = normalize_training_testing(X_train, X_test_list)

        end = timer()
        logging.debug(f'extract train and test: {end - begin:.2f} seconds') # 25 seconds for bluehex only

        save_hdf5(X_train, y_train, X_test_list, y_test_list, saved_data_path)

    logging.info(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
    logging.info(f'X_test_list: {[X_test.shape for X_test in X_test_list]}')
    logging.info(f'y_test_list: {[y_test.shape for y_test in y_test_list]}')
    return X_train, y_train, X_test_list, y_test_list


def get_family_training_meta(meta_df, test_begin_time, families_cnt, task='binary'):
    meta_avail = meta_df[meta_df.timestamp.str[:7] < test_begin_time]
    families = meta_avail['family'].tolist()
    c = Counter(families)
    selected_families = []
    for pair in c.most_common(families_cnt + 1): # benign + top N malicious families
        name, cnt = pair
        if task == 'binary':
            selected_families.append(name)
        else:
            if name == name: # do not add the benign class, # if name != name, then name = nan (which is benign)
                selected_families.append(name)
    meta_train = meta_avail[meta_avail['family'].isin(selected_families)]
    return meta_train


def prepare_testing_set(test_begin_time, test_end_time, general_data_folder, task='binary'):
    ''' prepare a list of testing sets from bluehex (each month as a testing set), fixed for all settings '''
    month_list = get_testing_month_list(test_begin_time, test_end_time)
    logging.debug(f'month_list: {month_list}')
    meta_df = load_meta('bluehex', general_data_folder)
    meta_test_list = []
    for month in month_list:
        meta_test = meta_df[meta_df.timestamp.str[:7] == month]
        logging.info(f'testing set month {month} # of families: {len(meta_test.family.unique()) - 1}')
        meta_test_list.append(meta_test)
    del meta_df
    logging.info(f'loading bluehex testing set ... ')
    X, y = load_npz_data('bluehex', general_data_folder)

    if task == 'binary':
        X_test_list = [np.array(X[m.index], dtype=np.float32) for m in meta_test_list]
        y_test_list = [np.array(y[m.index], dtype=np.float32) for m in meta_test_list]
    else:
        X_test_list = [np.array(X[m[~m.family.isna()].index], dtype=np.float32) for m in meta_test_list] # do not include benign
        y_test_list = [m[~m.family.isna()].family.to_numpy() for m in meta_test_list]
    del X, y
    logging.debug(f'X_test_list len: {len(X_test_list)}, y_test_list len: {len(y_test_list)}')
    logging.debug(f'X_test_list[0]: {X_test_list[0].shape}, y_test_list[0].shape: {y_test_list[0].shape}')
    logging.debug(f'y_test_list[0]: {Counter(y_test_list[0])}')
    return X_test_list, y_test_list


def normalize_training_testing(X_train, X_test_list):
    t1 = timer()
    X_train, X_test_list = normalize(X_train, X_test_list)
    t2 = timer()
    logging.debug(f'normalize time: {t2-t1:2f} seconds')
    return X_train, X_test_list


def save_hdf5(X_train, y_train, X_test_list, y_test_list, saved_data_path):
    t4 = timer()
    with h5py.File(saved_data_path, 'w') as hf:
        hf.create_dataset('X_train', data=X_train, compression="gzip")
        hf.create_dataset('y_train', data=y_train, compression="gzip")
        group = hf.create_group('test')
        for i, X_test, y_test in zip(range(len(X_test_list)), X_test_list, y_test_list):
            group.create_dataset(f'X_test_{i}', data=X_test, compression="gzip")
            group.create_dataset(f'y_test_{i}', data=y_test, compression="gzip")

    t5 = timer()
    logging.debug(f'save h5py compressed time: {t5-t4:.2f} seconds')


def load_npz_data(train_set, general_data_folder):
    data_path = os.path.join(general_data_folder, f'{train_set}.npz')
    npz_data = np.load(data_path)
    X, y = npz_data['X'], npz_data['y']
    return X, y


def load_meta(train_set, general_data_folder):
    meta_path = os.path.join(general_data_folder, f'{train_set}_metadata.csv')
    meta_df = pd.read_csv(meta_path, header=0)
    return meta_df


def normalize(X_train, X_test_list):
    scaler = MinMaxScaler()

    X_train_scale = scaler.fit_transform(X_train)
    logging.debug(f'X_train_scale: {X_train_scale.shape}')

    X_test_scale_list = []
    logging.debug(f'X_test_list: {len(X_test_list)}, X_test_list[0]: {X_test_list[0].shape}')
    for X_test in X_test_list:
        X_test_scale = scaler.transform(X_test)
        X_test_scale_list.append(X_test_scale)

    return X_train_scale, X_test_scale_list


def get_testing_month_list(begin_time, end_time):
    '''
    input: 2019-12, 2020-09
    output: [2019-12, 2020-01, 2020-02, ..., 2020-09]
    '''
    month_list = [begin_time]
    next_month = begin_time
    while(next_month != end_time):
        if next_month[5:] == '12': # 2019-12 -> 2020-01
            next_month = str(int(next_month[:4]) + 1) + '-01'
        elif next_month[5] == '0' and next_month[6] != '9': # 01 -> 02, 02 -> 03
            next_month = next_month[:5] + '0' + str(int(next_month[5:]) + 1)
        else: # 09 -> 10, 10 -> 11, 11 -> 12
            next_month = next_month[:5] + str(int(next_month[5:]) + 1)
        month_list.append(next_month)
    return month_list


def load_ember_drift_data(X, y, sample_month, test_month,
                          general_data_folder, saved_data_path):

    begin = timer()
    ''' use all samples from the first 1 month bluehex training set '''
    meta_df = load_meta('bluehex', general_data_folder)
    # meta_train = meta_df[meta_df.timestamp.str[:7] < test_begin_time]

    if type(sample_month) is list:
        meta_sample = meta_df[meta_df.timestamp.str[:7].isin(sample_month)]
        meta_test = meta_df[meta_df.timestamp.str[:7].isin(test_month)]
    else:
        meta_sample = meta_df[meta_df.timestamp.str[:7] == sample_month]
        meta_test = meta_df[meta_df.timestamp.str[:7] == test_month]

    X_sample_full = np.array(X[meta_sample.index], dtype=np.float32)
    y_sample_full = np.array(y[meta_sample.index], dtype=np.float32)

    X_test = np.array(X[meta_test.index], dtype=np.float32)
    y_test = np.array(y[meta_test.index], dtype=np.float32)

    end = timer()
    logging.debug(f'extract sample and test: {end - begin:.2f} seconds')
    logging.debug(f'saved_data_path: {saved_data_path}')

    logging.info(f'X_sample_full: {X_sample_full.shape}, y_sample_full: {y_sample_full.shape}')
    logging.info(f'X_test: {X_test.shape}, y_test: {y_test.shape}')

    return  X_sample_full, y_sample_full, X_test, y_test
