import keras
import numpy as np

# 1. Chargement du dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

NUM_CLASSES = 10
INPUT_SHAPE = x_train.shape[1:]  # (32, 32, 3)

# 2. Normalisation des pixels
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 3. Encodage One-Hot des labels
y_train = keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

print(f"Input data shape: {INPUT_SHAPE}")
print(f"y_train shape: {y_train.shape}")

# 4. Définition du modèle CNN classique

def build_basic_cnn(input_shape, num_classes):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_basic_cnn(INPUT_SHAPE, NUM_CLASSES)

if __name__ == "__main__":
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # 5. Entraînement du modèle
    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=10,
        validation_split=0.1
    )
    # 6. Évaluation sur le test set
    score = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {score[1]:.4f}")

# ---
# Section: Residual Block (ResNet)
def residual_block(x, filters, kernel_size=(3, 3), stride=1):
    y = keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same', activation='relu')(x)
    y = keras.layers.Conv2D(filters, kernel_size, padding='same')(y)
    if stride > 1:
        x = keras.layers.Conv2D(filters, (1, 1), strides=stride)(x)
    z = keras.layers.Add()([x, y])
    z = keras.layers.Activation('relu')(z)
    return z

# Example: Mini-ResNet architecture
input_layer = keras.Input(shape=INPUT_SHAPE)
x = residual_block(input_layer, 32)
x = residual_block(x, 64, stride=2)
x = residual_block(x, 64)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation='relu')(x)
out = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
resnet_model = keras.Model(inputs=input_layer, outputs=out)
resnet_model.summary()

# ---
# Section: Style Transfer (VGG16)
import matplotlib.pyplot as plt
from PIL import Image

def load_and_preprocess_image(path, size=(512, 512)):
    img = Image.open(path).resize(size)
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.vgg16.preprocess_input(img)
    return img

# Example usage (replace with your own images)
# content_image = load_and_preprocess_image('content.jpg')
# style_image = load_and_preprocess_image('style.jpg')

vgg = keras.applications.VGG16(include_top=False, weights='imagenet')
vgg.trainable = False
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

def create_extractor(model, style_layers, content_layers):
    outputs = [model.get_layer(name).output for name in style_layers + content_layers]
    return keras.Model(inputs=model.input, outputs=outputs)

extractor = create_extractor(vgg, style_layers, content_layers)
# The next step is to define a target image and optimize its pixels to minimize BOTH content loss AND style loss.
