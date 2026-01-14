import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Charger et prétraiter deux images (à adapter avec vos propres images)
def load_and_preprocess_image(path, size=(512, 512)):
    img = Image.open(path).resize(size)
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.vgg16.preprocess_input(img)
    return img

# Charger VGG16 pré-entraîné
vgg = keras.applications.VGG16(include_top=False, weights='imagenet')
vgg.trainable = False

content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

def create_extractor(model, style_layers, content_layers):
    outputs = [model.get_layer(name).output for name in style_layers + content_layers]
    return keras.Model(inputs=model.input, outputs=outputs)

extractor = create_extractor(vgg, style_layers, content_layers)

if __name__ == "__main__":
    # Charger les images
    content_image = load_and_preprocess_image('henri1.png')
    style_image = load_and_preprocess_image('henri2.png')

    # Afficher les images originales
    def show_image(img, title):
        img = img[0]
        img = img.copy()
        img = np.clip(img, 0, 255).astype('uint8')
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()

    show_image(keras.preprocessing.image.img_to_array(Image.open('henri1.png').resize((512, 512))), 'henri1 Image')
    show_image(keras.preprocessing.image.img_to_array(Image.open('henri2.png').resize((512, 512))), 'henri2 Image')

    # Style transfert simple (optimisation de l'image générée)
    generated_image = tf.Variable(content_image, dtype=tf.float32)

    # Fonctions de perte
    def gram_matrix(x):
        x = tf.squeeze(x)
        x = tf.transpose(x, (2, 0, 1))
        features = tf.reshape(x, (tf.shape(x)[0], -1))
        gram = tf.matmul(features, features, transpose_b=True)
        return gram / tf.cast(tf.size(x), tf.float32)

    optimizer = tf.optimizers.Adam(learning_rate=10.0)
    style_weight = 1e-2
    content_weight = 1e4

    for i in range(100):
        with tf.GradientTape() as tape:
            outputs = extractor(generated_image)
            style_outputs = outputs[:len(style_layers)]
            content_outputs = outputs[len(style_layers):]
            style_targets = extractor(style_image)[:len(style_layers)]
            content_targets = extractor(content_image)[len(style_layers):]
            style_loss = tf.add_n([
                tf.reduce_mean((gram_matrix(style_outputs[j]) - gram_matrix(style_targets[j])) ** 2)
                for j in range(len(style_layers))
            ])
            content_loss = tf.add_n([
                tf.reduce_mean((content_outputs[j] - content_targets[j]) ** 2)
                for j in range(len(content_layers))
            ])
            loss = style_weight * style_loss + content_weight * content_loss
        grads = tape.gradient(loss, generated_image)
        optimizer.apply_gradients([(grads, generated_image)])
        generated_image.assign(tf.clip_by_value(generated_image, 0, 255))
        if i % 20 == 0:
            print(f"Step {i}, Loss: {loss.numpy():.2f}")

    # Afficher et sauvegarder l'image générée
    output_img = generated_image.numpy()[0]
    output_img = np.clip(output_img, 0, 255).astype('uint8')
    plt.imshow(output_img)
    plt.title('Generated Image (Style Transfer)')
    plt.axis('off')
    plt.show()
    Image.fromarray(output_img).save('output_style_transfer.jpg')
    print("Image générée sauvegardée sous output_style_transfer.jpg")
