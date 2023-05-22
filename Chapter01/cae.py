import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

# Membangun Autoencoder
input_shape = (28, 28, 1)  # Dimensi input gambar MNIST (grayscale)

# Encoder
input_img = Input(shape=input_shape)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Membangun model Autoencoder
autoencoder = Model(input_img, decoded)

# Membangun model Encoder terpisah
encoder = Model(input_img, encoded)

# Compile dan melatih Autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Memuat dan mempersiapkan data MNIST
from keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

# Normalisasi dan reshape data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (*x_train.shape, 1))
x_test = np.reshape(x_test, (*x_test.shape, 1))

# Melatih Autoencoder
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

# Menggunakan Encoder untuk mendapatkan representasi ruang laten
latent_representation = encoder.predict(x_test)

# Contoh penggunaan ruang laten
sample_latent_vector = latent_representation[0]
reconstructed_image = autoencoder.predict(np.array([sample_latent_vector]))

# Menampilkan hasil rekonstruksi
import matplotlib.pyplot as plt

original_image = x_test[0].reshape(28, 28)
reconstructed_image = reconstructed_image.reshape(28, 28)

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image')

plt.show()
