"""
classifier.py
~~~~~~~

Functions for building a target classifier.

"""

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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential, model_from_json, load_model
from keras.optimizers import SGD, Adam
from keras.initializers import VarianceScaling
from keras.engine.topology import Layer, InputSpec
from keras.utils import np_utils, plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt

from cade.logger import LoggingCallback
import cade.utils as utils


class MLPClassifier(object):
    """a MLP classifier only for multi-class classification.
    """
    def __init__(self,
                 dims,
                 model_save_name,
                 dropout=0.2,
                 activation='relu',
                 verbose=1): # 1 print logs, 0 no logs.
        self.dims = dims  # e.g., [1347, 100, 30, 7]
        self.model_save_name = model_save_name
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
        x = Dense(self.dims[-1], activation='softmax',
                  name='clf_%d' % (n_stacks - 1))(x)
        output_tensor = x
        model = Model(inputs=input_tensor,
                      outputs=output_tensor, name='MLP')
        if self.verbose:
            logging.debug('MLP classifier summary: ' + str(model.summary()))
        return model

    def train(self, X_old, y_old,
              test_size=0.2,
              validation_split=0.1, # deprecated
              lr=0.001,  # learning rate
              batch_size=32,
              epochs=50,
              loss='categorical_crossentropy',
              class_weight=None,
              sample_weight=None,
              train_val_split=True, # whether to split train and validation
              retrain=True):
        """train a MLP classifier on old data (random split into train and validation),
        save the best model on highest acc on validation set.

        Arguments:
            X_old {np.ndarray} -- feature vectors for the old samples
            y_old {np.ndarray} -- groundtruth for the old samples
            x_new {np.ndarray} -- feature vectors for the new samples
            y_new {np.ndarray} -- groundtruth for the new samples
            retrain {boolean}  -- whether to train or use saved model.

        Returns:
            float -- the classifier's accuracy on the validation set.
        """
        if train_val_split: # used for normal training process
            x_train, x_val, y_train, y_val = train_test_split(X_old, y_old,
                                                              test_size=test_size,
                                                              random_state=42,
                                                              shuffle=True)

            # one-hot encoder. Assume multi-class classification as default.
            y_train_onehot = np_utils.to_categorical(y_train)
            y_val_onehot = np_utils.to_categorical(y_val)

            logging.debug(f'y_train onehot: {y_train_onehot.shape}')
            logging.debug(f'y_val onehot: {y_val_onehot.shape}')

            if retrain == True:
                model = self.build()

                # configure and train model.
                pretrain_optimizer = Adam(lr=lr)
                model.compile(loss=loss,
                            optimizer=pretrain_optimizer,
                            metrics=['accuracy'])

                utils.create_parent_folder(self.model_save_name)

                mcp_save = ModelCheckpoint(self.model_save_name,
                                        monitor='val_acc',
                                        save_best_only=True,
                                        save_weights_only=False,
                                        verbose=self.verbose,
                                        mode='max')
                if self.verbose:
                    callbacks = [mcp_save, LoggingCallback(logging.debug)]
                else:
                    callbacks = [mcp_save]
                history = model.fit(x_train, y_train_onehot,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_data=(x_val, y_val_onehot),
                                    verbose=self.verbose,
                                    class_weight=class_weight,
                                    sample_weight=sample_weight,
                                    callbacks=callbacks)
                val_acc = np.max(history.history['val_acc'])
                logging.getLogger('matplotlib.font_manager').disabled = True
                fig, ax = plt.subplots()
                ax.plot(history.history['loss'], '-b', label='Training')
                ax.plot(history.history['val_loss'], '--r', label='Testing')
                leg = ax.legend()
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.savefig(self.model_save_name + f'_lr{lr}_epoch{epochs}_loss.png', dpi=200)
                plt.clf()
            # evaluate on the test set
            K.clear_session() # to prevent load_model becomes slower and slower
            clf = load_model(self.model_save_name)
            val_score = clf.evaluate(x_val, y_val_onehot)
            val_acc = val_score[1]
            logging.info("MLP validation set %s: %.2f%%" %
                        (clf.metrics_names[1], val_acc*100))
        else: # do not split as train and validation, used for building the approximation_loose model
            y_old_onehot = np_utils.to_categorical(y_old)
            if retrain == True:
                model = self.build()
                # configure and train model.
                pretrain_optimizer = Adam(lr=lr)
                model.compile(loss=loss,
                            optimizer=pretrain_optimizer,
                            metrics=['accuracy'])

                utils.create_parent_folder(self.model_save_name)

                mcp_save = ModelCheckpoint(self.model_save_name,
                                            monitor='acc',
                                            save_best_only=True,
                                            save_weights_only=False,
                                            verbose=self.verbose,
                                            mode='max')
                if self.verbose:
                    callbacks = [mcp_save, LoggingCallback(logging.debug)]
                else:
                    callbacks = [mcp_save]
                history = model.fit(X_old, y_old_onehot,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    verbose=self.verbose,
                                    class_weight=class_weight,
                                    sample_weight=sample_weight,
                                    callbacks=callbacks)
                val_acc = np.max(history.history['acc'])
                logging.info("MLP training set acc: %.2f%%" % (val_acc*100))
                fig, ax = plt.subplots()
                ax.plot(history.history['loss'], '-b')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.savefig(self.model_save_name + f'_lr{lr}_epoch{epochs}_loss.png', dpi=200)
                plt.clf()
            K.clear_session() # to prevent load_model becomes slower and slower
            clf = load_model(self.model_save_name)
            val_score = clf.evaluate(X_old, y_old_onehot)
            val_acc = val_score[1]
        return val_acc


    def predict(self, X_new, y_new, dataset_name, newfamily, saved_cm_fig_path):
        K.clear_session()
        clf = load_model(self.model_save_name)

        # because new family can't one hot encoded (y_new has a label never shown in old data before.)
        y_pred = np.argmax(clf.predict(X_new), axis=1)
        new_acc = accuracy_score(y_new, y_pred)
        cm = confusion_matrix(y_new, y_pred)

        logging.info("MLP testing set %s: %.2f%%" %
                        (clf.metrics_names[1], new_acc * 100))
        logging.info(f'MLP confusion matrix: \n {cm}')

        utils.plot_confusion_matrix(cm, y_pred, y_new, dataset_name, newfamily, saved_cm_fig_path)

        return y_pred, new_acc


class RFClassifier(object):
    """RandomForest classifier wrapper.
    It internally supports multi-class classification. So don't need to one-hot encode the labels.
    """
    def __init__(self, rf_save_path, tree=100):
        self.rf_save_path = rf_save_path
        self.tree = tree

    def fit_and_predict(self,
                        X_old, y_old,
                        X_new, y_new,
                        dataset_name,
                        newfamily,
                        saved_cm_fig_path,
                        retrain,
                        test_size=0.2):
        x_train, x_test, y_train, y_test = train_test_split(X_old, y_old,
                                                            test_size=test_size,
                                                            random_state=42,
                                                            shuffle=True)

        if retrain == True:
            model = RandomForestClassifier(n_estimators=self.tree, random_state=0)
            model.fit(x_train, y_train)
            with open(self.rf_save_path, 'wb') as f:
                pickle.dump(model, f)

        with open(self.rf_save_path, 'rb') as f:
            model = pickle.load(f)
        y_test_pred = model.predict(x_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        logging.info("RF test samples acc: %.2f%%" % (test_acc*100))

        y_new_pred = model.predict(X_new)
        logging.debug('y_new_pred: ' + str(y_new_pred))
        new_acc = accuracy_score(y_new, y_new_pred)
        logging.info("RF new samples acc: %.2f%%" % (new_acc * 100))

        cm = confusion_matrix(y_new, y_new_pred)
        utils.plot_confusion_matrix(cm, y_new_pred, y_new, dataset_name, newfamily, saved_cm_fig_path)

        return y_new_pred, test_acc, new_acc

