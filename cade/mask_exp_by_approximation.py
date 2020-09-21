"""
mask_exp_by_approximation.py
~~~~~~~

Functions for mask explanation: why a sample is an drifting (approximation-based exp only).

exp = x * mask

Only use the target x to solve the mask, didn't use the x + noise (as the perturbation might not be a good choice).

"""

import os, sys
os.environ['PYTHONHASHSEED'] = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from numpy.random import seed
import random
random.seed(1)
seed(1)

from tensorflow import set_random_seed
set_random_seed(2)

from keras import backend as K
import tensorflow as tf

K.tensorflow_backend._get_available_gpus()

config = tf.ConfigProto()
# allocate as-needed
config.gpu_options.allow_growth = True
# only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# create a session with the above options specified
K.tensorflow_backend.set_session(tf.Session(config=config))

import sys
import logging

import numpy as np
import tensorflow as tf
from keras.losses import binary_crossentropy
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

class OptimizeExp(object):
    def __init__(self, input_shape, mask_shape, model, num_class, optimizer, initializer, lr, regularizer, model_file):
        '''
        Args:
            input_shape: the input shape for an input image
            mask_shape: the shape of the mask
            model:  the target model we want to explain
            num_class: number of distinct labels in the target model
            optimizer: tf.train.Optimizer() object
            initializer: initializer for the mask
            lr: learning rate
            regularizer: add regularization for the mask (to keep it as small as possible)
        '''

        self.model = model
        self.num_class = num_class
        self.lambda_1 = tf.placeholder(tf.float32) # placeholder is similar to cin of C++
        self.optimizer = optimizer(lr)
        self.initializer = initializer
        self.regularizer = regularizer
        self.build_opt_func(input_shape, mask_shape)
        self.model_file = model_file

    @staticmethod
    def elasticnet_loss(tensor):
        loss_l1 = tf.reduce_sum(tf.abs(tensor))
        loss_l2 = tf.sqrt(tf.reduce_sum(tf.square(tensor)))
        return loss_l1 + loss_l2

    def build_opt_func(self, input_shape, mask_shape):
        # use tf.variable_scope and tf.get_variable() to achieve "variable sharing"
        # AUTO_REUSE: we create variables if they do not exist, and return them otherwise
        with tf.variable_scope('mask', reuse=tf.AUTO_REUSE):
            self.mask = tf.get_variable('mask', shape=mask_shape, initializer=self.initializer)

        self.mask_reshaped = self.mask

        self.mask_normalized = tf.minimum(1.0, tf.maximum(self.mask_reshaped, 0.0))

        # get_input_at(node_index): Retrieves the input tensor(s) of a layer at a given node. node_index = 0 corresponds to the first time the layer was called.
        self.input = self.model.get_input_at(0)

        self.x_exp = self.input * self.mask_normalized # + self.fused_image * reverse_mask  # the explanation we are looking for, which contributes the most to the final prediction
        reverse_mask = tf.ones_like(self.mask_normalized) - self.mask_normalized
        self.x_remain = self.input * reverse_mask # + self.fused_image * self.mask_normalized

        '''
            because it's symbolic tensor with no actual value, so can't use model.predict()
            see: https://stackoverflow.com/questions/51515253/optimizing-a-function-involving-tf-kerass-model-predict-using-tensorflow-op
        '''
        self.output_exp = self.model(self.x_exp)  # self.x_exp is the input to the self.model
        self.output_remain = self.model(self.x_remain)

        self.y_target = tf.placeholder(tf.float32, shape=(None, self.num_class))
        self.loss_exp = tf.reduce_mean(binary_crossentropy(self.y_target, self.output_exp))
        self.loss_remain = tf.reduce_mean(binary_crossentropy(self.y_target, self.output_remain))

        if self.regularizer == 'l1':
            self.loss_reg_mask = tf.reduce_sum(tf.abs(self.mask_reshaped))
        elif self.regularizer == 'elasticnet':
            self.loss_reg_mask = self.elasticnet_loss(self.mask_reshaped)  # minimize mask
        elif self.regularizer == 'l2':
            self.loss_reg_mask = tf.sqrt(tf.reduce_sum(tf.square(self.mask_reshaped)))
        else:
            self.loss_reg_mask = tf.constant(0)

        self.loss = self.loss_exp - self.loss_remain + self.lambda_1 * self.loss_reg_mask

        # trainable variable
        self.var_train = tf.trainable_variables(scope='mask')  # only one trainable variable: mask/mask: 0

        # training function
        with tf.variable_scope('opt', reuse=tf.AUTO_REUSE):
            self.train_op = self.optimizer.minimize(self.loss, var_list=self.var_train)


    def fit_local(self, X, y, epochs, lambda_1, display_interval=10,
                  exp_acc_lowerbound=0.8, iteration_thredshold=1e-4, lambda_patience=100,
                  lambda_multiplier=1.5, early_stop_patience=10):
        """ explain a local prediction.

        Arguments:
            X {numpy array} -- A single image from the input.
            y {numpy vector} -- the probabilities of each label for X.
            epochs {int} -- [training epochs]
            lambda_1 {int} -- hyper parameter

        Keyword Arguments:
            display_interval {int} -- [display the loss values periodically] (default: {10})
            exp_acc_lowerbound {float} -- [the lowerbound of the accuracy by using only explanation part as features] (default: {0.8})
            iteration_thredshold {float} -- [if loss - loss_prev < threshold, early_stop + 1] (default: {1e-4})
            lambda_patience {int} -- [to work with lambda_up_counter or lambda_down_counter] (default: {100})
            lambda_multiplier {float} -- [if achieved lambda_patience, then increase/decrease lambda] (default: {1.5})
            early_stop_patience {int} -- [if loss didn't change much for X epoches, then stop] (default: {10})

        Returns:
            [numpy array] -- [the best mask we can found]
        """
        input_ = np.expand_dims(X, axis=0).reshape(1, -1)
        logging.debug(f'input_ shape: {input_.shape}')

        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=0)
        logging.debug(f'y shape: {y.shape}')

        loss_best = float('inf')
        loss_last = float('inf')
        loss_sparse_mask_best = float('inf')
        loss_sparse_mask_last = float('inf')

        mask_best = None
        early_stop_counter = 0
        lambda_up_counter = 0
        lambda_down_counter = 0

        # start training...
        with tf.Session() as sess:  # WARNING: it has to be like this, or the weights of the model could not be really loaded.
            sess.run(self.mask.initializer)
            sess.run(tf.variables_initializer(self.optimizer.variables()))
            sess.run(tf.initializers.global_variables())  # same as tf.global_variables_initializer()

            for step in range(epochs):
                feed_dict = {self.input: input_, self.y_target: y, self.lambda_1: lambda_1}

                '''
                debugging with tensorboard
                '''
                # logging.debug('debugging*******************')
                # writer = tf.summary.FileWriter("/tmp/mnist/", self.sess.graph)

                sess.run(self.train_op, feed_dict)

                self.model.load_weights(self.model_file, by_name=True)
                self.output_exp = self.model(self.x_exp)
                # logging.debug('current weights: ', self.model.get_layer('encoder_0').get_weights()[0][0][:5])

                '''
                debugging if the weights of the target model are correctly loaded
                '''
                # x_exp_value = sess.run(self.x_exp, feed_dict)
                # expected_exp = self.model.predict(x_exp_value)

                pred_exp = sess.run([self.output_exp], feed_dict)[0]
                [loss, loss_sparse_mask] = sess.run([self.loss, self.loss_reg_mask], feed_dict)

                acc_2 = accuracy_score(np.argmax(y, axis=1), np.argmax(pred_exp, axis=1))

                # check cost modification
                if acc_2 >= exp_acc_lowerbound:
                    lambda_up_counter += 1
                    if lambda_up_counter >= lambda_patience:
                        lambda_1 = lambda_1 * lambda_multiplier
                        lambda_up_counter = 0
                        logging.debug('Updating lambda1 to %.8f to %.8f'% (self.lambda_1))
                else:
                    lambda_down_counter += 1
                    if lambda_down_counter >= lambda_patience:
                        lambda_1 = lambda_1 / lambda_multiplier
                        lambda_down_counter = 0
                        logging.debug('Updating lambda1 to %.8f to %.8f'% (self.lambda_1))

                if (np.abs(loss - loss_last) < iteration_thredshold) or \
                        (np.abs(loss_sparse_mask - loss_sparse_mask_last) < iteration_thredshold):
                    early_stop_counter += 1

                if (acc_2 > exp_acc_lowerbound) and (early_stop_counter >= early_stop_patience):
                    logging.info('Reach the threshold and stop training at iteration %d/%d.' % (step+1, epochs))
                    mask_best = sess.run([self.mask_normalized])[0]
                    break

                loss_last = loss
                loss_sparse_mask_last = loss_sparse_mask

                if (step+1) % display_interval == 0:
                    mask = sess.run(self.mask)

                    if np.isnan(mask).any():
                        mask[np.isnan(mask)] = 1e-16
                        sess.run(self.mask.assign(mask))
                    feed_dict = {self.input: input_, self.y_target: y, self.lambda_1: lambda_1}
                    [pred_remain, pred_exp] = sess.run([self.output_remain, self.output_exp], feed_dict)
                    [loss, loss_exp, loss_remain, loss_sparse_mask] = \
                        sess.run([self.loss, self.loss_exp, self.loss_remain,
                                    self.loss_reg_mask], feed_dict)

                    acc_1 = accuracy_score(np.argmax(y, axis=1), np.argmax(pred_remain, axis=1))
                    acc_2 = accuracy_score(np.argmax(y, axis=1), np.argmax(pred_exp, axis=1))

                    # loss_sparse_mask: minimize mask

                    if loss_best > loss or loss_sparse_mask_best > loss_sparse_mask:
                        logging.debug(f'updating best loss from {loss_best} to {loss}')
                        logging.debug(f'updating best sparse mask loss from {loss_sparse_mask_best} to {loss_sparse_mask}')
                        logging.debug("Epoch %d/%d: loss = %.5f explanation_loss = %.5f remain_loss = %.5f "
                                        "mask_sparse_loss = %.5f acc_remain = %.5f acc_exp = %.5f"
                                        % (step+1, epochs, loss, loss_exp, loss_remain, loss_sparse_mask, acc_1, acc_2))
                        loss_best = loss
                        loss_sparse_mask_best = loss_sparse_mask
                        mask_best = sess.run([self.mask_normalized])[0]
        if mask_best is None:
            logging.info(f'did NOT find the best mask')

        return mask_best
