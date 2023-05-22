import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

# Membangun Autoencoder
input_dim = 784  # Jumlah piksel dalam gambar MNIST
latent_dim = 64  # Jumlah dimensi dalam ruang laten
hidden_layers = [512, 256, 128, 64]  # Jumlah unit dalam setiap lapisan tersembunyi

# Encoder
input_img = Input(shape=(input_dim,))
x = input_img
for units in hidden_layers:
    x = Dense(units, activation='relu')(x)
encoded = Dense(latent_dim, activation='relu')(x)

# Decoder
x = encoded
for units in hidden_layers[::-1]:
    x = Dense(units, activation='relu')(x)
decoded = Dense(input_dim, activation='sigmoid')(x)

# Membangun model Autoencoder
autoencoder = Model(input_img, decoded)

# Membangun model Encoder terpisah
encoder = Model(input_img, encoded)

# Compile dan melatih Autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Memuat dan mempersiapkan data MNIST
from keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

# Normalisasi dan flatten data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Melatih Autoencoder
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
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
