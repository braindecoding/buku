import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Reshape, Conv2D, Conv2DTranspose, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


# Membangun VAE
input_dim = 3092  # Jumlah fitur dalam data fMRI
latent_dim = 256  # Jumlah dimensi dalam ruang laten
output_shape = (28, 28, 1)  # Dimensi gambar (28x28 piksel, grayscale)
output_dim = 28 * 28  # Jumlah piksel dalam gambar

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
x = Dense(7 * 7 * 64, activation='relu')(z)
x = Reshape((7, 7, 64))(x)
x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)
decoded = Reshape(output_shape)(decoded)

# Membangun model VAE
encoder = Model(input_fMRI, z_mean)
decoder = Model(input_fMRI, decoded)
vae = Model(input_fMRI, decoder(input_fMRI))

# Loss function VAE
reconstruction_loss = mean_squared_error(K.flatten(input_fMRI), K.flatten(decoded))
reconstruction_loss *= input_dim
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)

# Membangun model supervised learning
output_fMRI = encoder(input_fMRI)
x = Dense(128, activation='relu')(output_fMRI)
output = Dense(output_dim, activation='sigmoid')(x)
output = Reshape(output_shape)(output)

model = Model(input_fMRI, output)

# Kompilasi model
model.compile(optimizer='adam', loss=mean_squared_error, metrics=['accuracy'])
model.summary()
# Memuat dan mempersiapkan data fMRI dan gambar
# Misalnya, x_train dan y_train adalah data fMRI dan gambar untuk pelatihan.
from lib import loaddata,plot
y_train, y_test,x_train,x_test=loaddata.Data28()
# Melatih model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluasi model
loss, accuracy = model.evaluate(x_test, y_test)

# Menggunakan model untuk memprediksi gambar dari data fMRI pengujian
predictions = model.predict(x_test)

# Menampilkan hasil prediksi
# Misalnya, Anda dapat melakukan visualisasi atau analisis pada hasil prediksi.

plot.tigaKolomGambar('Autoencoder CNN Denoising','Stimulus',y_test,'Rekonstruksi',predictions,'Recovery',predictions)
