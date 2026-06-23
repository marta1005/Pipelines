'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''

import time

import numpy as np

import tensorflow as tf

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel


class dkAE(BaseEstimator):
    """
    ## **Deep Kernelized Auto-Encoder**
    Defines an autoencoder crafted specifically to approximate kernel functions. Based on the paper:

    > Kampffmeyer, Michael, et al. "Deep kernelized autoencoders." Scandinavian conference on image analysis. Springer, Cham, 2017.

    and on the implementation:

    https://github.com/FilippoMB/TCK_AE

    ## -------------- WORK IN PROGRESS --------------

    #### Inputs
    ###### lambda_p
    Regularization parameter that gives the relative importance between the kernel loss and the reconstruction loss.
    ###### regloss_p
    Regularization parameter that multiplies the L2 loss.
    ###### hidden_size
    Number of neurons of the hidden layers
    ###### code_size
    Number of neurons of the code layer
    ###### hidden_layers
    Number of hidden layers before (and after) the code layer
    ###### kernel_function
    Kernel function to approximate. Current supported functions are:
     - "rbf": Radial Basis Function or Gaussian kernel
     - "lin": Linear Kernel. Mainly used for debug purposes
    ###### kernel_kwargs
    Dictionary of kwargs for the kernel function.
    ###### num_epochs
    Number of epochs to train the model.
    ###### epoch_verbosity_interval
    Number of epochs between verbose outputs.
    ###### batch_size
    Size of the training batch.
    ###### val_size
    Proportion of the training dataset to use as validation set.
    ###### random_state
    Seed for the RNG.
    ###### max_gradient_norm
    Max gradient norm to use in training.
    ###### learning_rate
    Learning rate for the optimizer.
    ###### tied_weights
    Whether the encoder and decoder weights are tied (are the same). This helps to reduce overfitting.
    ###### lin_dec
    Whether or not the decoder has linear activation functions.
    ###### model_name
    Name of the model to be stored.
    ###### verbosity
    Verbosity level.
    - If 0, the method is silent.
    - If 1, the method prints training time and total number of trainable parameters.
    - If > 1, the method prints training and validation metrics every `epoch_verbosity_interval` epochs.
    """

    def __init__(
        self,
        lambda_p=0.5,
        regloss_p=0,
        hidden_size=512,
        code_size=1000,
        hidden_layers=2,
        kernel_function="rbf",
        kernel_kwargs={"gamma": 1},
        num_epochs=100,
        epoch_verbosity_interval=10,
        batch_size=100,
        val_size=0.2,
        random_state=1234,
        max_gradient_norm=1,
        learning_rate=0.001,
        tied_weights=True,
        lin_dec=True,
        model_name="model.ckpt",
        verbosity=False,
    ):

        self.lambda_p = lambda_p
        self.regloss_p = regloss_p
        self.hidden_size = hidden_size
        self.code_size = code_size
        self.hidden_layers = hidden_layers
        self.kernel_function = kernel_function
        self.kernel_kwargs = kernel_kwargs
        self.num_epochs = num_epochs
        self.epoch_verbosity_interval = epoch_verbosity_interval
        self.batch_size = batch_size
        self.max_gradient_norm = max_gradient_norm
        self.learning_rate = learning_rate
        self.tied_weights = tied_weights
        self.lin_dec = lin_dec
        self.model_name = model_name
        self.verbosity = verbosity
        self.random_state = random_state
        self.val_size = val_size

    def fit(self, X, y):
        """
        ### **fit**
        Fits the model

        #### Inputs
        ###### X
        Contains the training data
        """

        X = np.array(X)

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as session:

            # ================= GRAPH =================

            X_train, X_val = train_test_split(X, test_size=self.val_size, random_state=self.random_state)

            input_length = X.shape[1]

            encoder_inputs = tf.compat.v1.placeholder(shape=(None, input_length), dtype=tf.float32, name="encoder_inputs")
            prior_K = tf.compat.v1.placeholder(shape=(None, None), dtype=tf.float32, name="prior_K")

            # ----- ENCODER -----
            we = []
            we.append(
                tf.Variable(
                    tf.random.uniform(
                        (input_length, self.hidden_size),
                        -1.0 / np.sqrt(input_length),
                        1.0 / np.sqrt(input_length),
                    )
                )
            )
            for i in range(self.hidden_layers - 1):
                we.append(
                    tf.Variable(
                        tf.random.uniform(
                            (self.hidden_size, self.hidden_size),
                            -1.0 / np.sqrt(self.hidden_size),
                            1.0 / np.sqrt(self.hidden_size),
                        )
                    )
                )
            we.append(
                tf.Variable(
                    tf.random.uniform(
                        (self.hidden_size, self.code_size),
                        -1.0 / np.sqrt(self.hidden_size),
                        1.0 / np.sqrt(self.hidden_size),
                    )
                )
            )

            be = [tf.Variable(tf.zeros([self.hidden_size])) for _ in range(self.hidden_layers)]
            be.append(tf.Variable(tf.zeros([self.code_size])))

            hidden = []
            hidden.append(tf.nn.tanh(tf.matmul(encoder_inputs, we[0]) + be[0]))
            for i in range(1, self.hidden_layers):
                hidden.append(tf.nn.tanh(tf.matmul(hidden[-1], we[i]) + be[i]))

            code = tf.nn.tanh(tf.matmul(hidden[-1], we[-1]) + be[-1], name="code")

            code_K = tf.tensordot(code, tf.transpose(code), axes=1)  # Kernel lineal

            # ----- DECODER -----

            if self.tied_weights:
                wd = [tf.transpose(we[-1 - i]) for i in range(len(we))]
            else:
                wd = []
                wd.append(
                    tf.Variable(
                        tf.random.uniform(
                            (self.code_size, self.hidden_size),
                            -1.0 / np.sqrt(self.code_size),
                            1.0 / np.sqrt(self.code_size),
                        )
                    )
                )
                for i in range(self.hidden_layers - 1):
                    wd.append(
                        wd2=tf.Variable(
                            tf.random.uniform(
                                (self.hidden_size, self.hidden_size),
                                -1.0 / np.sqrt(self.hidden_size),
                                1.0 / np.sqrt(self.hidden_size),
                            )
                        )
                    )
                wd.append(
                    tf.Variable(
                        tf.random.uniform(
                            (self.hidden_size, input_length),
                            -1.0 / np.sqrt(self.hidden_size),
                            1.0 / np.sqrt(self.hidden_size),
                        )
                    )
                )

            bd = [tf.Variable(tf.zeros([self.hidden_size])) for _ in range(self.hidden_layers)]
            bd.append(tf.Variable(tf.zeros([input_length])))

            if self.lin_dec:
                hidden_d = []
                hidden_d.append(tf.matmul(code, wd[0]) + bd[0])
                for i in range(1, self.hidden_layers):
                    hidden_d.append(tf.matmul(hidden_d[-1], wd[i]) + bd[i])
            else:
                hidden_d = []
                hidden_d.append(tf.nn.tanh(tf.matmul(code, wd[0]) + bd[0]))
                for i in range(1, self.hidden_layers):
                    hidden_d.append(tf.nn.tanh(tf.matmul(hidden_d[-1], wd[i]) + bd[i]))

            dec_out = tf.matmul(hidden_d[-1], wd[-1]) + bd[-1]

            # ----- LOSS -----

            # kernel alignment loss with normalized Frobenius norm
            code_K_norm = code_K / tf.norm(code_K, ord="fro", axis=[-2, -1])
            prior_K_norm = prior_K / tf.norm(prior_K, ord="fro", axis=[-2, -1])
            k_loss = tf.norm(code_K_norm - prior_K_norm, ord="fro", axis=[-2, -1])
            saved_k_loss = tf.identity(k_loss, name="k_loss")

            # reconstruction loss
            parameters = tf.compat.v1.trainable_variables()
            optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
            reconstruct_loss = tf.compat.v1.losses.mean_squared_error(labels=dec_out, predictions=encoder_inputs)
            saved_reconstruct_loss = tf.identity(reconstruct_loss, name="reconstruct_loss")

            # L2 loss
            reg_loss = 0
            for tf_var in tf.compat.v1.trainable_variables():
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

            loss = (1 - self.lambda_p) * reconstruct_loss + self.regloss_p * reg_loss + self.lambda_p * k_loss
            saved_loss = tf.identity(loss, name="loss")

            # Calculate and clip gradients
            gradients = tf.gradients(loss, parameters)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            update_step = optimizer.apply_gradients(zip(clipped_gradients, parameters))

            session.run(tf.compat.v1.global_variables_initializer())

            # trainable parameters count
            total_parameters = 0
            for variable in tf.compat.v1.trainable_variables():
                shape = variable.get_shape()
                variable_parametes = 1
                for dim in shape:
                    variable_parametes *= dim.value
                total_parameters += variable_parametes

            if self.verbosity:
                print("Total parameters: {}".format(total_parameters))

            # ================= TRAINING =================

            # initialize training variables
            time_tr_start = time.time()
            max_batches = X_train.shape[0] // self.batch_size
            max_batches_val = X_val.shape[0] // self.batch_size
            loss_track = []
            reconstruct_loss_track = []
            kloss_track = []
            min_vs_loss = np.infty
            saver = tf.compat.v1.train.Saver()

            try:

                history = {"epochs": [], "train_losses": [], "val_losses": []}

                for ep in range(self.num_epochs):

                    kloss_epoch = []
                    loss_epoch = []
                    reconstruct_loss_epoch = []

                    # shuffle training data
                    idx = np.random.permutation(X_train.shape[0])

                    for batch in range(max_batches):
                        indices = idx[(batch) * self.batch_size : (batch + 1) * self.batch_size]

                        K_tr_s = self.__get_kernel_matrix(X_train[indices, :], self.kernel_function, **self.kernel_kwargs)
                        train_data_s = X_train[indices, :]

                        fdtr = {encoder_inputs: train_data_s, prior_K: K_tr_s}
                        (
                            _,
                            train_loss,
                            train_reconstruct_loss,
                            train_kloss,
                        ) = session.run([update_step, loss, reconstruct_loss, k_loss], fdtr)
                        loss_track.append(train_loss)
                        reconstruct_loss_track.append(train_reconstruct_loss)
                        kloss_track.append(train_kloss)

                        reconstruct_loss_epoch.append(train_reconstruct_loss)
                        kloss_epoch.append(train_kloss)
                        loss_epoch.append(train_loss)

                    if ep % self.epoch_verbosity_interval == 0:
                        if self.verbosity > 1:
                            print("Ep: {}".format(ep), time.time() - time_tr_start)
                        loss_epoch_avg = sum(loss_epoch) / len(loss_epoch)
                        reconstruct_loss_epoch_avg = sum(reconstruct_loss_epoch) / len(reconstruct_loss_epoch)
                        kloss_epoch_avg = sum(kloss_epoch) / len(kloss_epoch)

                        idx = np.random.permutation(X_val.shape[0])

                        reconstruct_loss_val = []
                        kloss_val = []
                        loss_val = []

                        for batch in range(max_batches_val):
                            indices = idx[(batch) * self.batch_size : (batch + 1) * self.batch_size]
                            K_vs = self.__get_kernel_matrix(X_val[indices, :], self.kernel_function, **self.kernel_kwargs)
                            val_data_s = X_val[indices, :]

                            fdvs = {encoder_inputs: val_data_s, prior_K: K_vs}

                            (
                                outvs,
                                lossvs,
                                reconstruct_lossvs,
                                klossvs,
                                vs_code_K,
                            ) = session.run([dec_out, loss, reconstruct_loss, k_loss, code_K], fdvs)

                            reconstruct_loss_val.append(reconstruct_lossvs)
                            kloss_val.append(klossvs)
                            loss_val.append(lossvs)

                        reconstruct_loss_val_avg = sum(reconstruct_loss_val) / len(reconstruct_loss_val)
                        kloss_val_avg = sum(kloss_val) / len(kloss_val)
                        loss_val_avg = sum(loss_val) / len(loss_val)

                        if self.verbosity > 1:
                            print(
                                "VS loss=%.3f, reconstruct_loss=%.3f, k_loss=%.3f -- TR loss=%.3f, reconstruct_loss=%.3f, k_loss=%.3f"
                                % (
                                    loss_val_avg,
                                    reconstruct_loss_val_avg,
                                    kloss_val_avg,
                                    loss_epoch_avg,
                                    reconstruct_loss_epoch_avg,
                                    kloss_epoch_avg,
                                )
                            )

                        if loss_val_avg < min_vs_loss:
                            min_vs_loss = loss_val_avg
                            tf.compat.v1.add_to_collection("encoder_inputs", encoder_inputs)
                            tf.compat.v1.add_to_collection("code", code)
                            tf.compat.v1.add_to_collection("dec_out", dec_out)
                            tf.compat.v1.add_to_collection("reconstruct_loss", reconstruct_loss)
                            tf.compat.v1.add_to_collection("loss", saved_loss)
                            tf.compat.v1.add_to_collection("k_loss", saved_k_loss)
                            tf.compat.v1.add_to_collection("reconstruct_loss", saved_reconstruct_loss)
                            tf.compat.v1.add_to_collection("prior_K", prior_K)
                            save_path = saver.save(session, self.model_name)

                        history["epochs"].append(ep)
                        history["train_losses"].append(
                            [
                                loss_epoch_avg,
                                reconstruct_loss_epoch_avg,
                                kloss_epoch_avg,
                            ]
                        )
                        history["val_losses"].append([loss_val_avg, reconstruct_loss_val_avg, kloss_val_avg])

                    if ep == range(self.num_epochs)[-1]:
                        loss_epoch_avg = sum(loss_epoch) / len(loss_epoch)
                        reconstruct_loss_epoch_avg = sum(reconstruct_loss_epoch) / len(reconstruct_loss_epoch)
                        kloss_epoch_avg = sum(kloss_epoch) / len(kloss_epoch)

                        idx = np.random.permutation(X_val.shape[0])

                        reconstruct_loss_val = []
                        kloss_val = []
                        loss_val = []

                        for batch in range(max_batches_val):
                            indices = idx[(batch) * self.batch_size : (batch + 1) * self.batch_size]
                            K_vs = self.__get_kernel_matrix(X_val[indices, :], self.kernel_function, **self.kernel_kwargs)
                            val_data_s = X_val[indices, :]

                            fdvs = {encoder_inputs: val_data_s, prior_K: K_vs}

                            (
                                outvs,
                                lossvs,
                                reconstruct_lossvs,
                                klossvs,
                                vs_code_K,
                            ) = session.run([dec_out, loss, reconstruct_loss, k_loss, code_K], fdvs)

                            reconstruct_loss_val.append(reconstruct_lossvs)
                            kloss_val.append(klossvs)
                            loss_val.append(lossvs)

                        reconstruct_loss_val_avg = sum(reconstruct_loss_val) / len(reconstruct_loss_val)
                        kloss_val_avg = sum(kloss_val) / len(kloss_val)
                        loss_val_avg = sum(loss_val) / len(loss_val)

                        if loss_val_avg < min_vs_loss:
                            min_vs_loss = loss_val_avg
                            tf.compat.v1.add_to_collection("encoder_inputs", encoder_inputs)
                            tf.compat.v1.add_to_collection("code", code)
                            tf.compat.v1.add_to_collection("dec_out", dec_out)
                            tf.compat.v1.add_to_collection("reconstruct_loss", reconstruct_loss)
                            tf.compat.v1.add_to_collection("loss", saved_loss)
                            tf.compat.v1.add_to_collection("k_loss", saved_k_loss)
                            tf.compat.v1.add_to_collection("reconstruct_loss", saved_reconstruct_loss)
                            tf.compat.v1.add_to_collection("prior_K", prior_K)
                            save_path = saver.save(session, self.model_name)

                        history["epochs"].append(ep)
                        history["train_losses"].append(
                            [
                                loss_epoch_avg,
                                reconstruct_loss_epoch_avg,
                                kloss_epoch_avg,
                            ]
                        )
                        history["val_losses"].append([loss_val_avg, reconstruct_loss_val_avg, kloss_val_avg])

            except KeyboardInterrupt:
                print("training interrupted")

            except Exception as e:
                print(e)

            time_tr_end = time.time()
            if self.verbosity:
                print("Tot training time: %.2f" % (time_tr_end - time_tr_start))

            self.fitted_ = True

            return self

    def __get_kernel_matrix(self, X, kernel_function, **kwargs):
        allowed_kernels = {"rbf": rbf_kernel, "lin": linear_kernel}

        if kernel_function not in allowed_kernels:
            raise AttributeError("%s kernel not available" % (kernel_function))

        kernel = allowed_kernels[kernel_function]

        return kernel(X, **kwargs)

    def transform(self, X):
        """
        ### **transform**
        Transform the given X data into code space.

        #### Inputs
        ###### X
        Contains the data to be transformed.

        #### Outputs
        ###### hat_code
        Contains the transformed data into code space.
        """

        check_is_fitted(self, attributes="fitted_")

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as session:
            saver = tf.compat.v1.train.import_meta_graph(self.model_name + ".meta")
            saver.restore(session, self.model_name)

            graph = tf.get_default_graph()
            encoder_inputs = graph.get_tensor_by_name("encoder_inputs:0")
            code = graph.get_tensor_by_name("code:0")

            hat_code = session.run(code, {encoder_inputs: X})

        return hat_code

    def score(self, X, model_name=None):
        """
        ### **score**
        Scores the model.

        #### Inputs
        ###### X
        Data which with the model will be scored.

        #### Outputs
        ###### score
        Score of the model
        """

        check_is_fitted(self, attributes="fitted_")

        if model_name is None:
            model_name = self.model_name

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as session:
            saver = tf.compat.v1.train.import_meta_graph(model_name + ".meta")
            saver.restore(session, model_name)

            graph = tf.get_default_graph()
            encoder_inputs = graph.get_tensor_by_name("encoder_inputs:0")
            prior_K = graph.get_tensor_by_name("prior_K:0")
            code = graph.get_tensor_by_name("code:0")
            loss_tensor = graph.get_tensor_by_name("loss:0")

            idx = np.random.permutation(X.shape[0])
            max_indices = X.shape[0] if X.shape[0] < self.batch_size else X.shape[0]
            indices = idx[0:max_indices]

            kernel_matrix = self.__get_kernel_matrix(X[indices, :], self.kernel_function, **self.kernel_kwargs)
            loss = session.run(loss_tensor, {encoder_inputs: X[indices, :], prior_K: kernel_matrix})

        return np.exp(-loss)

    def loss_function(self, X, model_name=None):
        """
        ### **loss_function**
        Loss function used to train the model

        #### Inputs
        ###### X
        Data with which the loss is calculated

        #### Outpus
        ###### loss
        Total loss of the model
        ###### reconstruct_loss
        Reconstruct loss. Measures the distance between the reconstructed vectors and their originals.
        ###### k_loss
        Kernel loss. Measures the distance between the Kernel matrix and the approximated one.
        """
        check_is_fitted(self, attributes="fitted_")

        if model_name is None:
            model_name = self.model_name

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as session:
            saver = tf.compat.v1.train.import_meta_graph(model_name + ".meta")
            saver.restore(session, model_name)

            graph = tf.get_default_graph()
            encoder_inputs = graph.get_tensor_by_name("encoder_inputs:0")
            prior_K = graph.get_tensor_by_name("prior_K:0")
            code = graph.get_tensor_by_name("code:0")
            loss_tensor = graph.get_tensor_by_name("loss:0")
            k_loss_tensor = graph.get_tensor_by_name("k_loss:0")
            reconstruct_loss_tensor = graph.get_tensor_by_name("reconstruct_loss:0")

            idx = np.random.permutation(X.shape[0])
            max_indices = X.shape[0] if X.shape[0] < self.batch_size else X.shape[0]
            indices = idx[0:max_indices]

            kernel_matrix = self.__get_kernel_matrix(X[indices, :], self.kernel_function, **self.kernel_kwargs)
            k_loss = session.run(k_loss_tensor, {encoder_inputs: X[indices, :], prior_K: kernel_matrix})
            reconstruct_loss = session.run(
                reconstruct_loss_tensor,
                {encoder_inputs: X[indices, :], prior_K: kernel_matrix},
            )
            loss = session.run(loss_tensor, {encoder_inputs: X[indices, :], prior_K: kernel_matrix})

        return loss, reconstruct_loss, k_loss
