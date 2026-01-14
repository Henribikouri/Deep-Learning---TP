import tensorflow as tf
import keras

# Bloc résiduel simplifié
def residual_block(x, filters, kernel_size=(3, 3), stride=1):
    y = keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same', activation='relu')(x)
    y = keras.layers.Conv2D(filters, kernel_size, padding='same')(y)
    if stride > 1:
        x = keras.layers.Conv2D(filters, (1, 1), strides=stride)(x)
    z = keras.layers.Add()([x, y])
    z = keras.layers.Activation('relu')(z)
    return z

# Mini-architecture avec 3 blocs résiduels
INPUT_SHAPE = (32, 32, 3)
input_layer = keras.Input(shape=INPUT_SHAPE)
x = residual_block(input_layer, 32)
x = residual_block(x, 64, stride=2)
x = residual_block(x, 64)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation='relu')(x)
out = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=input_layer, outputs=out)

if __name__ == "__main__":
    print("Mini-ResNet model summary:")
    model.summary()

    # Chargement du dataset CIFAR-10
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=10,
        validation_split=0.1
    )
    score = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {score[1]:.4f}")
