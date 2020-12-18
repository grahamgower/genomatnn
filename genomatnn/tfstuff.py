"""
All tensorflow/keras stuff is separated out into this file, to permit
configuration via environment variables before this file is imported.
When tensorflow is imported, things like OMP_NUM_THREADS become fixed
and are no longer configurable.
"""

import os
import logging

import numpy as np

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    # Mute annoying messages that most users won't/can't do anything about:
    #     Your CPU supports instructions that this TensorFlow binary was not
    #     compiled to use: ...
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if "KMP_AFFINITY" not in os.environ:
    # See also https://github.com/tensorflow/tensorflow/issues/29354
    os.environ["KMP_AFFINITY"] = "granularity=fine,noverbose,compact,1,0"

import tensorflow as tf  # noqa
from tensorflow.keras import backend as K  # noqa
from tensorflow.keras import models, initializers  # noqa
from tensorflow.keras.layers import (  # noqa
    Input,
    Conv2D,
    Dense,
    MaxPooling2D,
    BatchNormalization,
    Lambda,
    Concatenate,
    Flatten,
    LeakyReLU,
)

logger = logging.getLogger(__name__)


def tf_config(n_threads):
    """
    Set fixed number of CPU threads in CPU-only operation.
    Wow, tensorflow, why is this all so hard?
    """
    if n_threads > 0:
        if len(tf.config.list_physical_devices("GPU")) > 0:
            logger.warning("Configuring tensorflow for CPU-only use.")
        # tf.config.set_visible_devices([], "GPU") doesn't leave GPUs
        # available for other processes. Setting CUDA_VISIBLE_DEVICES="" is the
        # only reliable way to stop tensorflow from locking GPUs.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    logger.debug(f"Configuring tensorflow to use {n_threads} CPUs")
    tf.config.threading.set_intra_op_parallelism_threads(n_threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    omp_num_threads = int(os.environ.get("OMP_NUM_THREADS", 0))
    if omp_num_threads != n_threads:
        # It's too late to set OMP_NUM_THREADS, all we can do now is warn the user.
        logger.warning(
            "Attempt to configure tensorflow to use a fixed number of "
            "threads will likely be ineffective without also setting "
            "the OMP_NUM_THREADS environment variable."
        )


def ConvLayer(n_filt, filt_size, strides=(1, 1)):
    stddev = 1.0 / (filt_size[0] * filt_size[1])
    return Conv2D(
        n_filt,
        filt_size,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=stddev),
    )


def basic_cnn(
    input_shape,
    output_shape,
    n_conv,
    n_conv_filt,
    filt_size_x,
    filt_size_y,
    n_dense,
    dense_size,
):
    assert output_shape == 1, "Only binary classification is supported for now."
    main_input = Input(shape=input_shape)
    x = main_input

    for _ in range(n_conv):
        x = BatchNormalization()(x)
        x = ConvLayer(n_conv_filt, (filt_size_x, filt_size_y), strides=(2, 2))(x)
        x = LeakyReLU()(x)

    x = Flatten()(x)

    for _ in range(n_dense):
        x = BatchNormalization()(x)
        x = Dense(dense_size)(x)
        x = LeakyReLU()(x)

    # Output layers.
    x = BatchNormalization()(x)
    output = Dense(output_shape, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=main_input, outputs=output, name="basic_cnn")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def InvariantLayer(axis=2, name="invariant"):
    """
    This layer applies a function that is invariant to permutation along
    the specified axis.
    """
    # TODO: The mean function used here is likely not sufficient.
    return Lambda(lambda z: K.mean(z, axis=axis, keepdims=True), name=name)


def permutation_invariant_cnn(
    input_shape, output_shape, n_conv, n_conv_filt, filt_size, n_dense, dense_size
):
    """
    A permutation invariant CNN, with individuals from all populations poooled
    together when the permutation invariant function is applied.

    This follows the approach of Chan et al. 2018.
    https://doi.org/10.1101/267211
    """
    assert output_shape == 1, "Only binary classification is supported for now."
    main_input = Input(shape=input_shape)
    x = main_input

    for _ in range(n_conv):
        x = BatchNormalization()(x)
        x = ConvLayer(n_conv_filt, (filt_size, 1))(x)
        x = LeakyReLU()(x)

    x = InvariantLayer()(x)

    for _ in range(n_conv):
        x = BatchNormalization()(x)
        x = ConvLayer(n_conv_filt, (filt_size, 1))(x)
        x = LeakyReLU()(x)

    x = Flatten()(x)

    for _ in range(n_dense):
        x = BatchNormalization()(x)
        x = Dense(dense_size)(x)
        x = LeakyReLU()(x)

    # Output layers.
    x = BatchNormalization()(x)
    output = Dense(output_shape, activation="sigmoid", name="output")(x)

    model = models.Model(
        inputs=main_input, outputs=output, name="permutation_invariant_cnn"
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def per_population_permutation_invariant_cnn(
    input_shape,
    output_shape,
    pop_starts,
    pop_ends,
    n_conv,
    n_conv_filt,
    filt_size,
    n_dense,
    dense_size,
):
    """
    A permutation invariant CNN, with the permutation invariant function
    applied separately to each population.
    """
    assert output_shape == 1, "Only binary classification is supported for now."
    assert len(pop_starts) == len(pop_ends)
    main_input = Input(shape=input_shape)

    poptensors = []
    for i, (idx0, idx1) in enumerate(zip(pop_starts, pop_ends)):
        # Partition main_input into population-specific tensors.
        x = Lambda(lambda z, a=idx0, b=idx1: z[:, :, a:b, :], name=f"slice_{i}")(
            main_input
        )
        poptensors.append(x)
        idx0 = idx1

    for i, x in enumerate(poptensors):
        for _ in range(n_conv):
            x = BatchNormalization()(x)
            x = ConvLayer(n_conv_filt, (filt_size, 1))(x)
            x = LeakyReLU()(x)
        # Individual-level permutation invariance.
        x = InvariantLayer(name=f"invariant_ind_{i}")(x)
        poptensors[i] = x

    x = Concatenate(axis=2)(poptensors)

    for _ in range(n_conv):
        x = BatchNormalization()(x)
        x = ConvLayer(n_conv_filt, (filt_size, 1))(x)
        x = LeakyReLU()(x)

    x = Flatten()(x)

    for _ in range(n_dense):
        x = BatchNormalization()(x)
        x = Dense(dense_size)(x)
        x = LeakyReLU()(x)

    # Output layers.
    x = BatchNormalization()(x)
    output = Dense(output_shape, activation="sigmoid", name="output")(x)

    model = models.Model(
        inputs=main_input,
        outputs=output,
        name="per_population_permutation_invariant_cnn",
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def get_nn_model(nn_model, input_shape, output_shape, params):
    if nn_model == "cnn":
        model = basic_cnn(input_shape, output_shape, **params)
    else:
        raise ValueError(f"Unknown neutral network model {nn_model}")
    return model


def train(conf, train_x, train_y, val_x, val_y):
    tf_config(conf.parallelism)
    # Add "channels" dimension of size 1. We don't use channels.
    train_x = np.expand_dims(train_x, axis=-1)
    val_x = np.expand_dims(val_x, axis=-1)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = get_nn_model(conf.nn_model, train_x.shape[1:], 1, conf.nn_model_params)
    if conf.verbose:
        model.summary()

    train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    val_data = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    train_data = train_data.batch(conf.train_batch_size)
    val_data = val_data.batch(conf.train_batch_size)

    # Set the autoshart policy to make tensorflow shutup.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    train_data = train_data.with_options(options)
    val_data = val_data.with_options(options)

    history = model.fit(
        train_data,
        epochs=conf.train_epochs,
        validation_data=val_data,
        verbose=1,
    )
    logger.info(f"Saving model to {conf.nn_hdf5_file}")
    model.save(conf.nn_hdf5_file)
    return history


def predict(conf, model, input_data):
    """
    Wrapper around model.predict() that explicitly sets the AutoShartPolicy.
    """
    data = tf.data.Dataset.from_tensor_slices(input_data)
    data = data.batch(conf.train_batch_size)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    data = data.with_options(options)
    return model.predict(data)
