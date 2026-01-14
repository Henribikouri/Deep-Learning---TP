import mlflow
import mlflow.tensorflow  # ou mlflow.keras (les deux fonctionnent)
import tensorflow as tf
from mlflow.models.signature import infer_signature
import numpy as np

# Chargement des données MNIST
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Séparation en train, validation, test (TP2 Exercice 1)
# Utiliser 90% pour train, 10% pour val
x_val = x_train_full[54000:]
y_val = y_train_full[54000:]
x_train = x_train_full[:54000]
y_train = y_train_full[:54000]

# Normalisation
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Variables pour les paramètres
EPOCHS = 5
BATCH_SIZE = 128
DROPOUT_RATE = 0.2
L2_REG = 0.001  # Pour régularisation L2

# Fonction pour créer le modèle avec régularisation et BatchNorm (TP2 Exercice 2 et 4)
def create_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG)),
        tf.keras.layers.BatchNormalization(),  # TP2 Exercice 4
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    return model

# Optimiseurs à comparer (TP2 Exercice 3)
optimizers = {
    'Adam': 'adam',
    'SGD_with_momentum': tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'RMSprop': 'rmsprop'
}

# Boucle pour comparer les optimiseurs
for opt_name, optimizer in optimizers.items():
    with mlflow.start_run(run_name=f"TP2_Optimizer_{opt_name}"):
        # Créer le modèle
        model = create_model()
        
        # Compiler avec l'optimiseur actuel
        model.compile(optimizer=optimizer,
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        
        # Enregistrer les paramètres
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("dropout_rate", DROPOUT_RATE)
        mlflow.log_param("l2_reg", L2_REG)
        mlflow.log_param("optimizer", opt_name)
        
        # Entraînement avec validation_data (TP2 Exercice 1)
        history = model.fit(x_train, y_train,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=(x_val, y_val),
                            verbose=2)
        
        # Évaluation sur test
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        
        # Enregistrer les métriques
        mlflow.log_metric("test_accuracy", float(test_acc))
        mlflow.log_metric("test_loss", float(test_loss))
        mlflow.log_metric("train_accuracy", float(history.history['accuracy'][-1]))
        mlflow.log_metric("val_accuracy", float(history.history['val_accuracy'][-1]))
        mlflow.log_metric("train_loss", float(history.history['loss'][-1]))
        mlflow.log_metric("val_loss", float(history.history['val_loss'][-1]))
        
        # Signature pour le modèle
        sample_input = x_test[:10]
        sample_output = model.predict(sample_input)
        signature = infer_signature(sample_input, sample_output)
        
        # Enregistrer le modèle
        mlflow.keras.log_model(model, "mnist-model", signature=signature)
        
        print(f"Optimizer {opt_name}: Test accuracy: {test_acc:.4f}")

# Sauvegarder le modèle final (avec Adam par exemple)
final_model = create_model()
final_model.compile(optimizer='adam',
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
final_model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val), verbose=0)
final_model.save("mnist_model.h5")
print("Modèle final sauvegardé sous mnist_model.h5")
