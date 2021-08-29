# -*- coding: utf-8 -*-

"""
utils.py
~~~~~~~~

Helper functions for setting up the environment and parsing args, etc.

"""

import os
os.environ['PYTHONHASHSEED'] = '0'
from numpy.random import seed
import random
random.seed(1)
seed(1)

import sys
import logging
import argparse
import pickle
import json
import numpy as np
import pandas as pd
import smtplib
import traceback
import lightgbm as lgb

from pprint import pformat
from collections import Counter
from email.mime.text import MIMEText
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)


def create_parent_folder(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))


def parse_multiple_dataset_args():
    """Parse the command line configuration for a particular run.

    Raises:
        ValueError: if the tree value for RandomForest is negative.

    Returns:
        argparse.Namespace -- a set of parsed arguments.
    """
    p = argparse.ArgumentParser()

    p.add_argument('--task', default='binary', choices=['binary', 'multiclass'],
                    help='Whether to perform binary classification or multi-class classification.')

    p.add_argument('--training-set',
                   help='Which extra dataset to use as training. Blue Hexagon first 3 months is the default training set.')

    p.add_argument('--diversity', choices=['no', 'size', 'family', 'timestamp', 'timestamp_part', 'legacy', 'packed', 'family_fixed_size'],
                   help='Which diversity metric to use in the training set: size, timestamp, family, packed. \
                        "no" means the original setting: use the 3 months of bluehex dataset as the training set.')


    p.add_argument('--setting-name', help='name for this particular setting, for saving corresponding data, model, and results')

    p.add_argument('-c', '--classifier', choices=['rf', 'gbdt', 'mlp'],
                    help='The classifier used for binary classification or multi-class classification')

    p.add_argument('--testing-time',
                   help='The beginning time and ending time (separated by comma) for a particular testing set (bluehex data)')

    p.add_argument('--quiet', default=1, type=int, choices=[0, 1], help='whether to print DEBUG logs or just INFO')


    p.add_argument('--retrain', type=int, choices=[0, 1], default=0,
                   help='Whether to retrain the classifier, default NO.')

    p.add_argument('--seed', type=int, default=42, help='random seed for training and validation split.')


    # sub-arguments for the family (binary) and family_fixed_size (binary) diversity and multi-class classification
    p.add_argument('--families', type=int, help='add top N families from the first three months of bluehex.')

    # sub-arguments for the MLP classifier.
    p.add_argument('--mlp-hidden',
                   help='The hidden layers of the MLP classifier, e.g.,: "2400-1200-1200", would make the architecture as 2381-2400-1200-1200-2')
    p.add_argument('--mlp-batch-size', default=32, type=int,
                   help='MLP classifier batch_size.')
    p.add_argument('--mlp-lr', default=0.001, type=float,
                   help='MLP classifier Adam learning rate.')
    p.add_argument('--mlp-epochs', default=50, type=int,
                   help='MLP classifier epochs.')
    p.add_argument('--mlp-dropout', default=0.2, type=float,
                   help='MLP classifier Droput rate.')

    # sub-arguments for the RandomForest classifier.
    p.add_argument('--tree',
                   type=int,
                   default=100,
                   help='The n_estimators of RandomForest classifier when --classifier = "rf"')

    args = p.parse_args()

    if args.tree < 0:
        raise ValueError('invalid tree value')

    return args


def get_model_dims(model_name, input_layer_num, hidden_layer_num, output_layer_num):
    """convert hidden layer arguments to the architecture of a model (list)
    Arguments:
        model_name {str} -- 'MLP' or 'Contrastive AE'.
        input_layer_num {int} -- The number of the features.
        hidden_layer_num {str} -- The '-' connected numbers indicating the number of neurons in hidden layers.
        output_layer_num {int} -- The number of the classes.
    Returns:
        [list] -- List represented model architecture.
    """
    try:
        if '-' not in hidden_layer_num:
            dims = [input_layer_num, int(hidden_layer_num), output_layer_num]
        else:
            hidden_layers = [int(dim) for dim in hidden_layer_num.split('-')]
            dims = [input_layer_num]
            for dim in hidden_layers:
                dims.append(dim)
            dims.append(output_layer_num)
        logging.debug(f'{model_name} dims: {dims}')
    except:
        logging.error(f'get_model_dims {model_name}\n{traceback.format_exc()}')
        sys.exit(-1)

    return dims


def dump_json(data, output_dir, filename, overwrite=True):
    dump_data('json', data, output_dir, filename, overwrite)


def dump_data(protocol, data, output_dir, filename, overwrite=True):
    file_mode = 'w' if protocol == 'json' else 'wb'
    fname = os.path.join(output_dir, filename)
    logging.info(f'Dumping data to {fname}...')
    if overwrite or not os.path.exists(fname):
        with open(fname, file_mode) as f:
            if protocol == 'json':
                json.dump(data, f, indent=4)
            else:
                # pickle.dump(data, f)
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(filename):
    with open(filename, 'r') as f:
        d = json.load(f) # dict
    return d


def parse_drift_args():
    """Parse the command line configuration for a particular run.

    Raises:
        ValueError: if the tree value for RandomForest is negative.

    Returns:
        argparse.Namespace -- a set of parsed arguments.
    """
    p = argparse.ArgumentParser()

    p.add_argument('--task', default='binary', choices=['binary', 'multiclass'],
                    help='Whether to perform binary classification or multi-class classification.')

    p.add_argument('--setting-name', help='name for this particular setting, for saving corresponding data, model, and results')

    p.add_argument('-c', '--classifier', choices=['rf', 'gbdt', 'mlp'],
                    help='The classifier used for binary classification or multi-class classification')

    p.add_argument('--testing-time',
                   help='The beginning time and ending time (separated by comma) for a particular testing set (bluehex data)')

    p.add_argument('--month-interval', type=int, default=1, help='specify how many months for sampling.')

    # sub-arguments for the family (binary) and family_fixed_size (binary) diversity and multi-class classification
    p.add_argument('--families', type=int, help='add top N families from the first three months of bluehex.')

    p.add_argument('--quiet', default=1, type=int, choices=[0, 1], help='whether to print DEBUG logs or just INFO')

    p.add_argument('--retrain', type=int, choices=[0, 1], default=0,
                   help='Whether to retrain the classifier, default NO.')

    p.add_argument('--sample-ratio', default=0.01, type=float, help='how many samples to add back to the training set for retraining to combat concept drift.')
    p.add_argument('--ember-ratio', default=0.3, type=float, help='how many Ember samples to train Transcend / CADE.')

    p.add_argument('--seed', default=1, type=int, help='random seed for the random experiment')

    args = p.parse_args()
    return args


def normalize_sample_month(X_train_origin, X_sample_full):
    scaler = MinMaxScaler()
    X_train_scale = scaler.fit_transform(X_train_origin)
    X_sample_scale = scaler.transform(X_sample_full)
    return X_sample_scale


def get_ember_sample_idx_by_pred_proba(X_sample_full, y_sample_full,
                                       sample_month_str, args,
                                       test_begin_time, REPORT_FOLDER,
                                       baseline_model_path, fpr_threshold):
    SUB_REPORT_FOLDER = os.path.join(REPORT_FOLDER, 'intermediate')
    os.makedirs(SUB_REPORT_FOLDER, exist_ok=True)
    report_path = os.path.join(SUB_REPORT_FOLDER, f'{args.classifier}_{test_begin_time}_ranked_proba_sample_{sample_month_str}.csv')
    if os.path.exists(report_path):
        df = pd.read_csv(report_path)
        sorted_sample_idx = df['idx'].to_numpy()
    else:
        clf = lgb.Booster(model_file=baseline_model_path)
        y_sample_pred = clf.predict(X_sample_full)
        y_sample_prob = np.array([1 - t if t < 0.5 else t for t in y_sample_pred])
        y_sample_pred_label = np.array(y_sample_pred > fpr_threshold, dtype=np.int)

        sorted_sample_idx = np.argsort(y_sample_prob)

        with open(report_path, 'w') as f:
            f.write(f'idx,real,pred,proba\n')
            for i in sorted_sample_idx:
                f.write(f'{i},{y_sample_full[i]},{y_sample_pred_label[i]},{y_sample_prob[i]}\n')
    return sorted_sample_idx
