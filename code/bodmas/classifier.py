import os
os.environ['PYTHONHASHSEED'] = '0'
from numpy.random import seed
import random
random.seed(1)
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from keras import backend as K
import tensorflow as tf

import sys
import json
import warnings
import logging
import pickle
from datetime import datetime
import traceback
import seaborn as sns

from collections import Counter, OrderedDict
from timeit import default_timer as timer

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential, model_from_json, load_model
from keras.optimizers import Adam
from keras.initializers import VarianceScaling
from keras.engine.topology import Layer, InputSpec
from keras.utils import np_utils, plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt

from bodmas.logger import LoggingCallback
import bodmas.utils as utils



def train_model(X_train, y_train, SAVED_MODEL_PATH, params={}, redo_flag=False):
    """
    DEPRECATED
    Train the LightGBM model from the blue hexagon dataset.
    """
    if not redo_flag:
        if os.path.exists(SAVED_MODEL_PATH):
            logging.info("Loading pre-trained LightGBM model...")
            lgbm_model = lgb.Booster(model_file=SAVED_MODEL_PATH)
        else:
            logging.info("Training LightGBM model...")
            # update params
            params.update({"application": "binary"})

            # Train, it would use all 24 CPUs
            lgbm_dataset = lgb.Dataset(X_train, y_train)
            lgbm_model = lgb.train(params, lgbm_dataset)
            lgbm_model.save_model(SAVED_MODEL_PATH)
    else:
        logging.info("Training LightGBM model...")
        # update params
        params.update({"application": "binary"})

        # Train, it would use all 24 CPUs
        lgbm_dataset = lgb.Dataset(X_train, y_train)
        lgbm_model = lgb.train(params, lgbm_dataset)
        lgbm_model.save_model(SAVED_MODEL_PATH)

    return lgbm_model



class GBDTClassifier(object):
    def __init__(self, saved_model_path):
        self.saved_model_path = saved_model_path

    def train(self, X_train, y_train, task, families_cnt, retrain, params={}):
        if not os.path.exists(self.saved_model_path) and retrain == False:
            retrain = True
        if retrain:
            begin = timer()
            logging.info('Training LightGBM model...')
            # params.update({'application': 'binary'}) # not needed since 'application' is an alias to 'objective'
            if task != 'binary':
                params.update({'objective': 'multiclass',
                               'metric': 'multi_logloss', # alias for softmax
                               'num_class': families_cnt})

            # Train, it would use all 24 CPUs
            lgbm_dataset = lgb.Dataset(X_train, y_train)
            lgbm_model = lgb.train(params, lgbm_dataset)
            lgbm_model.save_model(self.saved_model_path)
            end = timer()
            logging.info(f'Training LightGBM finished, time: {end - begin:.1f} seconds')
        else:
            logging.info("Loading pre-trained LightGBM model...")
            lgbm_model = lgb.Booster(model_file=self.saved_model_path)

        return lgbm_model


class MLPClassifier(object):
    def __init__(self,
                 saved_model_path,
                 dims,
                 dropout,
                 activation='relu',
                 verbose=1): # 1 print logs, 0 no logs.
        self.dims = dims  # e.g., [2381, 1024, 1024, 512, 2]
        self.saved_model_path = saved_model_path
        self.act = activation
        self.dropout = dropout
        self.verbose = verbose

    def build(self):
        # build a MLP model with Keras functional API
        n_stacks = len(self.dims) - 1
        input_tensor = Input(shape=(self.dims[0],), name='input')
        x = input_tensor
        for i in range(n_stacks - 1):
            x = Dense(self.dims[i + 1],
                      activation=self.act, name='clf_%d' % i)(x)
            if self.dropout > 0:
                x = Dropout(self.dropout, seed=42)(x)

        x = Dense(self.dims[-1], activation='sigmoid',
                  name='clf_%d' % (n_stacks - 1))(x)
        output_tensor = x
        model = Model(inputs=input_tensor,
                      outputs=output_tensor, name='MLP')
        if self.verbose:
            logging.debug('MLP classifier summary: ' + str(model.summary()))
        return model

    def train(self, X_train, y_train, X_val, y_val, retrain,
              lr=0.001,  # learning rate
              batch_size=32,
              epochs=50,
              loss= 'binary_crossentropy', #'categorical_crossentropy',
              class_weight=None):
        """train a MLP classifier on training set and validation set,
        save the best model on highest acc on validation set.
        Arguments:
            X_train {np.ndarray} -- feature vectors for the old samples
            y_train {np.ndarray} -- groundtruth for the old samples
            retrain {boolean}  -- whether to train or use saved model.
        Returns:
            float -- the classifier's accuracy on the validation set.
        """
        if not os.path.exists(self.saved_model_path) and retrain == False:
            retrain = True

        if retrain:
            begin = timer()
            logging.info('Training MLP model...')
            model = self.build()

            # configure and train model.
            pretrain_optimizer = Adam(lr=lr)
            model.compile(loss=loss,
                          optimizer=pretrain_optimizer,
                          metrics=['accuracy'])

            utils.create_parent_folder(self.saved_model_path)

            mcp_save = ModelCheckpoint(self.saved_model_path,
                                        monitor='val_acc',
                                        save_best_only=True,
                                        save_weights_only=False,
                                        verbose=self.verbose,
                                        mode='max')
            if self.verbose:
                callbacks = [mcp_save, LoggingCallback(logging.debug)]
            else:
                callbacks = [mcp_save]
            history = model.fit(X_train, y_train, # change to y_train_onehot for mulit-class
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=(X_val, y_val), # change to y_val_onehot for multi-class
                                verbose=self.verbose,
                                class_weight=class_weight,
                                callbacks=callbacks)
            end = timer()
            logging.info(f'Training MLP finished, time: {end - begin:.1f} seconds')
            fig, ax = plt.subplots()
            ax.plot(history.history['loss'], '-b', label='Training')
            ax.plot(history.history['val_loss'], '--r', label='Testing')
            leg = ax.legend()
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.savefig(self.saved_model_path + f'_loss.png', dpi=72)
            plt.clf()

        K.clear_session() # to prevent load_model becomes slower and slower
        clf = load_model(self.saved_model_path)
        return clf


class RFClassifier(object):
    ''' RandomForest classifier wrapper.
        It internally supports multi-class classification. So don't need to one-hot encode the labels.
    '''
    def __init__(self, saved_model_path, tree=100):
        self.saved_model_path = saved_model_path
        self.tree = tree

    def train(self, X_train, y_train, retrain):
        if not os.path.exists(self.saved_model_path) and retrain == False:
            retrain = True
        if retrain:
            begin = timer()
            logging.info('Training RandomForest model...')
            model = RandomForestClassifier(n_estimators=self.tree, random_state=0, n_jobs=-1, verbose=3)
            model.fit(X_train, y_train)
            with open(self.saved_model_path, 'wb') as f:
                pickle.dump(model, f)
            end = timer()
            logging.info(f'Training RandomForest finished, time: {end - begin:.1f} seconds')
        else:
            logging.info('Loading pre-trained RandomForest model...')
            with open(self.saved_model_path, 'rb') as f:
                model = pickle.load(f)

        return model
