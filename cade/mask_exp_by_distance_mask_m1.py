"""
Our method: use m * m1 as explanation, only when m = m1 = 1, feature is important.
Explaining drift: minimize the difference between a drift $x$ and an in distribution centroid $c$ by swapping a
                small proportion of features.
Loss function:  \min E_{m \sim Bern(p)} ||f(x * (1 - m * m1) + (1-x)*(m * m1)), f(centroid)||_2 + \lambda * ||m * m1||_{1+2}
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

import warnings
warnings.filterwarnings('ignore')


class OptimizeExp(object):
    def __init__(self, batch_size, mask_shape, latent_dim, model, optimizer, initializer, lr,
                 regularizer, temp, normalize_choice, use_concrete, model_file):
        """ Explaining drift: minimize the difference between a drift $x$ and an in distribution centroid $c$ by swapping a
                            small proportion of features.
        :param batch_size: training batch size.
        :param mask_shape: shape of the mask.
        :param latent_dim: shape of the latent representation.
        :param model: trained encoder model.
        :param optimizer: optimizer tf class.
        :param initializer: initializer tf class.
        :param lr: initial learning rate.
        :param regularizer: choice of regularizer.
        :param temp: temperature.
        :param normalize_choice: how to normalize the variable ('sigmoid', 'tanh', 'clip').
        :param use_concrete: use concrete distribution or not.
        """

        self.model = model
        self.lambda_1 = tf.placeholder(tf.float32) # placeholder is similar to cin of C++
        self.optimizer = optimizer(lr)
        self.initializer = initializer
        self.regularizer = regularizer
        self.batch_size = batch_size
        self.temp = temp
        self.normalize_choice = normalize_choice
        self.use_concrete = use_concrete
        self.build_opt_func(mask_shape, latent_dim)
        self.model_file = model_file

    @staticmethod
    def concrete_transformation(p, mask_shape, batch_size, temp=1.0 / 10.0):
        """ Use concrete distribution to approximate binary output.
        :param p: Bernoulli distribution parameters.
        :param temp: temperature.
        :param batch_size: size of samples.
        :return: approximated binary output.
        """
        epsilon = np.finfo(float).eps  # 1e-16

        unif_noise = tf.random_uniform(shape=(batch_size, mask_shape[0]),
                                       minval=0, maxval=1)
        reverse_theta = tf.ones_like(p) - p
        reverse_unif_noise = tf.ones_like(unif_noise) - unif_noise

        appro = tf.log(p + epsilon) - tf.log(reverse_theta + epsilon) + \
                tf.log(unif_noise) - tf.log(reverse_unif_noise)
        logit = appro / temp

        return tf.sigmoid(logit)

    @staticmethod
    def elasticnet_loss(tensor):
        loss_l1 = tf.reduce_sum(tf.abs(tensor))
        loss_l2 = tf.sqrt(tf.reduce_sum(tf.square(tensor)))
        return loss_l1 + loss_l2

    def build_opt_func(self, mask_shape, latent_dim):

        # define and prepare variables.
        with tf.variable_scope('p', reuse=tf.AUTO_REUSE):
            self.p = tf.get_variable('p', shape=mask_shape, initializer=self.initializer)

        ## normalize variables
        if self.normalize_choice == 'sigmoid':
            logging.debug('Using sigmoid normalization.')
            self.p_normalized = tf.sigmoid(self.p)
        elif self.normalize_choice == 'tanh':
            logging.debug('Using tanh normalization.')
            self.p_normalized = (tf.tanh(self.p + 1)) / (2 + tf.keras.backend.epsilon())
        else:
            logging.debug('Using clip normalization.')
            self.p_normalized = tf.minimum(1.0, tf.maximum(self.p, 0.0))

        ## discrete variables to continuous variables.
        if self.use_concrete:
            self.mask = self.concrete_transformation(self.p_normalized, mask_shape, self.batch_size, self.temp)
        else:
            self.mask = self.p_normalized
        self.reverse_p = tf.ones_like(self.p_normalized) - self.p_normalized

        # get input and reverse input.
        self.input = self.model.get_input_at(0)
        self.reverse_x = tf.placeholder(tf.float32, shape=(None, mask_shape[0]))

        self.m1 = tf.placeholder(tf.float32, shape=mask_shape)
        self.reverse_mask = tf.ones_like(self.mask) - self.mask * self.m1
        # if flip their feature value, it would be closer to the centroid
        self.x_exp = self.input * self.reverse_mask + self.reverse_x * self.mask * self.m1
        self.centroid = tf.placeholder(tf.float32, shape=(None, latent_dim))
        self.output_exp = self.model(self.x_exp)

        # l2 norm distance.
        self.loss_exp = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.output_exp - self.centroid), axis=1)))

        if self.regularizer == 'l1':
            self.loss_reg_mask = tf.reduce_sum(tf.abs(self.p_normalized * self.m1))
        elif self.regularizer == 'elasticnet':
            self.loss_reg_mask = self.elasticnet_loss(self.p_normalized * self.m1)  # minimize mask
        elif self.regularizer == 'l2':
            self.loss_reg_mask = tf.sqrt(tf.reduce_sum(tf.square(self.p_normalized * self.m1)))
        else:
            self.loss_reg_mask = tf.constant(0)

        self.loss = self.loss_exp + self.lambda_1 * self.loss_reg_mask

        # trainable variable
        self.var_train = tf.trainable_variables(scope='p')

        # training function
        with tf.variable_scope('opt', reuse=tf.AUTO_REUSE):
            self.train_op = self.optimizer.minimize(self.loss, var_list=self.var_train)

    def fit_local(self, X, m1, centroid, closest_to_centroid_sample, num_sync, num_changed_fea, epochs, lambda_1,
                  display_interval=10, exp_loss_lowerbound=0.17, iteration_threshold=1e-4, lambda_patience=100,
                  lambda_multiplier=1.5, early_stop_patience=10):

        """ fit explanation.
        :param X: input sample
        :param centroid: low dimsion centroid.
        :param num_sync: num of sync sample.
        :param num_changed_fea: num of changed features.
        :param epochs: training epochs.
        :param lambda_1: sparsity loss penalty term.
        :param display_interval: print information interval.
        :param exp_loss_lowerbound: penalty term update threshold.
        :param iteration_threshold: early stop count threshold.
        :param lambda_patience: lambda update patience
        :param lambda_multiplier: lambda update multiplier
        :param early_stop_patience: early stop wait patience
        :return: The solved explanation mask.
        """

        # assuming the shape of X (p,)
        # swap a small number of features from the target drift sample to synthesize new drift sample,
        # so we can have more drift samples for the concrete distribution gumbel trick.
        sync_idx = np.random.choice(X.shape[0], (num_sync, num_changed_fea))
        sync_x = np.repeat(X[None, :], num_sync, axis=0).reshape(num_sync, X.shape[0])
        for i in range(num_sync):
            sync_x[i, sync_idx[i]] = 1 - X[sync_idx[i]]

        input_ = np.vstack((X, sync_x))
        logging.debug(f'input_ shape: {input_.shape}')

        sync_lowd = self.model.predict(input_)
        dis = np.square((sync_lowd - centroid))
        dis = np.mean(np.sqrt(np.sum(dis, axis=1)))
        logging.debug(f'x_target + synthesized sample average distance to centroid: {dis}')

        if input_.shape[0] % self.batch_size != 0:
            num_batch = (input_.shape[0] // self.batch_size) + 1
        else:
            num_batch = (input_.shape[0] // self.batch_size)
        idx = np.arange(input_.shape[0])

        loss_best = float('inf')
        loss_sparse_mask_best = float('inf')
        loss_last = float('inf')
        loss_sparse_mask_last = float('inf')

        mask_best = None
        early_stop_counter = 0
        lambda_up_counter = 0
        lambda_down_counter = 0

        # start training...
        with tf.Session() as sess:
            sess.run(tf.initializers.global_variables())  # same as tf.global_variables_initializer()
            self.model.load_weights(self.model_file, by_name=True)

            for step in range(epochs):
                loss_tmp = []
                loss_exp_tmp = []
                loss_sparse_mask_tmp = []
                for i in range(num_batch):
                    feed_dict = {self.input: input_[idx[i * self.batch_size:min((i + 1) * self.batch_size, input_.shape[0])],],
                                 self.lambda_1: lambda_1, self.centroid: centroid[None, ],
                                 self.reverse_x: closest_to_centroid_sample[None, ],
                                 self.m1: m1}
                    sess.run(self.train_op, feed_dict)
                    # NOTE: we don't need to load weights every batch. this is really time consuming. 5x time
                    # self.model.load_weights(self.model_file, by_name=True)
                    [loss, loss_sparse_mask, loss_exp] = sess.run([self.loss, self.loss_reg_mask,
                                                                        self.loss_exp], feed_dict)
                    loss_tmp.append(loss)
                    loss_exp_tmp.append(loss_exp)
                    loss_sparse_mask_tmp.append(loss_sparse_mask)

                loss = sum(loss_tmp) / len(loss_tmp)
                loss_exp = sum(loss_exp_tmp) / len(loss_exp_tmp)
                loss_sparse_mask = sum(loss_sparse_mask_tmp) / len(loss_sparse_mask_tmp)

                if loss_exp <= exp_loss_lowerbound:
                        lambda_up_counter += 1
                        if lambda_up_counter >= lambda_patience:
                            lambda_1 = lambda_1 * lambda_multiplier
                            lambda_up_counter = 0
                else:
                    lambda_down_counter += 1
                    if lambda_down_counter >= lambda_patience:
                        lambda_1 = lambda_1 / lambda_multiplier
                        lambda_down_counter = 0

                if (np.abs(loss - loss_last) < iteration_threshold) or \
                        (np.abs(loss_sparse_mask - loss_sparse_mask_last) < iteration_threshold):
                    early_stop_counter += 1

                if (loss_exp <= exp_loss_lowerbound) and (early_stop_counter >= early_stop_patience):
                    logging.info('Reach the threshold and stop training at iteration %d/%d.' % (step + 1, epochs))
                    mask_best = sess.run([self.p_normalized])[0]
                    break

                if (step+1) % display_interval == 0:
                    mask = sess.run(self.p)
                    if np.isnan(mask).any():
                        mask[np.isnan(mask)] = 1e-16
                        sess.run(self.mask.assign(mask))

                    if loss_best > loss or loss_sparse_mask_best > loss_sparse_mask:
                        logging.debug(f'updating best loss from {loss_best} to {loss}')
                        logging.debug(f'updating best sparse mask loss from {loss_sparse_mask_best} to {loss_sparse_mask}')
                        logging.debug("Epoch %d/%d: loss = %.5f explanation_loss = %.5f "
                                        "mask_sparse_loss = %.5f "
                                        % (step+1, epochs, loss, loss_exp, loss_sparse_mask))
                        loss_best = loss
                        loss_sparse_mask_best = loss_sparse_mask
                        mask_best = sess.run([self.p_normalized])[0]

                loss_last = loss
                loss_sparse_mask_last = loss_sparse_mask

        if mask_best is None:
            logging.info(f'did NOT find the best mask')

        return mask_best
