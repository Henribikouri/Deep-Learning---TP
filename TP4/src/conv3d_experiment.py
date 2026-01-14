"""conv3d_experiment.py

Small demonstration of a Conv3D block and MLflow logging for TP4.
This script builds a tiny Conv3D model, simulates training on random
volumetric data, and logs model config and a dummy metric to MLflow.
"""
import mlflow
import numpy as np
import keras


def simple_conv3d_block(input_shape=(32, 32, 32, 1)):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = keras.layers.MaxPool3D((2, 2, 2))(x)

    x = keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPool3D((2, 2, 2))(x)

    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs, outputs, name='Conv3D_Small')


def run_experiment():
    mlflow.set_experiment('3D_Volumetric_Analysis')
    with mlflow.start_run(run_name='Conv3D_Baseline'):
        model = simple_conv3d_block()
        model_config = model.to_json()
        mlflow.log_dict({'model_config': model_config}, 'artifacts/model_architecture.json')
        mlflow.log_param('optimizer', 'adam')
        mlflow.log_param('filters_start', 16)

        # Simulate training: create random data and compute a dummy loss
        X = np.random.rand(4, 32, 32, 32, 1).astype('float32')
        y = (np.random.rand(4, 1) > 0.5).astype('float32')
        # Instead of training (slow), evaluate a forward pass
        preds = model.predict(X)
        # Log a fake metric
        mlflow.log_metric('simulated_val_loss', float(((preds - y) ** 2).mean()))

    print('Conv3D experiment logged to MLflow (simulated).')


if __name__ == '__main__':
    run_experiment()
