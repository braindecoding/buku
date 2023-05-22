import numpy as np
from keras.layers import Input, Dense, Reshape, Conv2D, Conv2DTranspose, Flatten,Lambda
from keras.models import Model
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.datasets import mnist

# Membangun VAE
input_dim = 3092  # Jumlah fitur dalam data fMRI
output_shape = (28, 28, 1)  # Bentuk output gambar (misalnya, 28x28 grayscale)
latent_dim = 64  # Jumlah dimensi dalam ruang laten

# Encoder
input_fMRI = Input(shape=(input_dim,))
x = Dense(512, activation='relu')(input_fMRI)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

# Latent Distribution
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# Decoder
x = Dense(128, activation='relu')(z)
x = Dense(256, activation='relu')(x)
x = Dense(512, activation='relu')(x)
decoded = Dense(np.prod(output_shape), activation='sigmoid')(x)
decoded = Reshape(output_shape)(decoded)

# Membangun model VAE
encoder = Model(input_fMRI, z_mean)
decoder = Model(input_fMRI, decoded)
vae = Model(input_fMRI, decoder(input_fMRI))

# Loss function VAE
reconstruction_loss = binary_crossentropy(K.flatten(input_fMRI), K.flatten(decoded)) #alternative bisa menggunakan mean_squared_error
reconstruction_loss *= np.prod(output_shape)
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# Memuat dan mempersiapkan data fMRI
# Misalnya, x_train dan x_test adalah data fMRI yang telah dinormalisasi dan direshape sesuai dengan dimensi yang tepat.

# Melatih VAE
vae.fit(x_train, epochs=10, batch_size=32, validation_data=(x_test, None))

# Menggunakan Decoder untuk menghasilkan gambar dari data fMRI pengujian
decoded_images = decoder.predict(x_test)

# Menampilkan hasil rekonstruksi
# Misalnya, Anda dapat melakukan visualisasi pada decoded_images.

