"""unet_segmentation.py

Minimal, self-contained U-Net implementation with Dice and IoU metrics
and a tiny example training loop on dummy data. Intended for TP4.

Usage (example):
    python src/unet_segmentation.py

This script creates a small U-Net, compiles it with a sensible loss and
logs a tiny run to MLflow. It uses random data so it is fast and only
for verifying the pipeline.
"""
import json
import numpy as np
import mlflow
import tensorflow as tf
import keras
from keras import backend as K


def conv_block(input_tensor, num_filters):
    x = keras.layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    return x


def build_unet(input_shape=(128, 128, 1)):
    inputs = keras.Input(shape=input_shape)

    # ENCODER
    c1 = conv_block(inputs, 32)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 128)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)

    # BRIDGE / BOTTLENECK
    b = conv_block(p3, 256)

    # DECODER
    u1 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b)
    u1 = keras.layers.Concatenate()([u1, c3])
    d1 = conv_block(u1, 128)

    u2 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d1)
    u2 = keras.layers.Concatenate()([u2, c2])
    d2 = conv_block(u2, 64)

    u3 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(d2)
    u3 = keras.layers.Concatenate()([u3, c1])
    d3 = conv_block(u3, 32)

    outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(d3)

    return keras.Model(inputs=inputs, outputs=outputs, name='UNet_Small')


def dice_coeff(y_true, y_pred, smooth=1.0):
    # Avoid using K.flatten (not present in some Keras backend versions).
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coeff(y_true, y_pred)


def iou_metric(y_true, y_pred, smooth=1.0):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def compile_model(model, lr=1e-3):
    opt = keras.optimizers.Adam(learning_rate=lr)
    # Use binary crossentropy + dice loss to handle class imbalance in segmentation
    def combined_loss(y_true, y_pred):
        bce = keras.losses.binary_crossentropy(y_true, y_pred)
        return bce + dice_loss(y_true, y_pred)

    model.compile(optimizer=opt,
                  loss=combined_loss,
                  metrics=[dice_coeff, iou_metric])
    return model


def train_on_dummy_data(epochs=1, batch_size=4):
    input_shape = (128, 128, 1)
    model = build_unet(input_shape)
    compile_model(model)

    # Create tiny dummy dataset
    N = 8
    X = np.random.rand(N, *input_shape).astype(np.float32)
    # Create sparse binary masks to emulate small targets
    Y = (np.random.rand(N, *input_shape) > 0.95).astype(np.float32)

    mlflow.set_experiment('UNet_Segmentation_TP4')
    run_name = f"UNet_adam_combinedloss_epochs{epochs}"
    with mlflow.start_run(run_name=run_name):
        # Log model architecture as a JSON artifact
        model_config = model.to_json()
        mlflow.log_param('optimizer', 'adam')
        mlflow.log_param('loss', 'bce+1-dice')
        mlflow.log_param('input_shape', input_shape)
        mlflow.log_param('filters_start', 32)

        # Fit for a tiny number of epochs to validate pipeline
        history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=1)

        # Log final metrics
        for k, v in history.history.items():
            # history.history contains lists per metric
            mlflow.log_metric(k, float(v[-1]))

        # Save weights as artifact
        weights_path = 'unet_dummy_weights.weights.h5'
        model.save_weights(weights_path)
        mlflow.log_artifact(weights_path)

    print('Done. A tiny training run was logged to MLflow (if MLflow is reachable).')


if __name__ == '__main__':
    # Run a tiny verification training
    train_on_dummy_data(epochs=1, batch_size=4)
