import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# Membangun DGMM
input_dim = 1000  # Jumlah fitur dalam data fMRI
latent_dim = 64  # Jumlah dimensi dalam ruang laten
hidden_layers = [512, 256, 128]  # Jumlah unit dalam setiap lapisan tersembunyi
num_components = 5  # Jumlah komponen dalam model campuran

# Encoder
input_img = Input(shape=(input_dim,))
x = input_img
for units in hidden_layers:
    x = Dense(units, activation='relu')(x)
encoded = Dense(latent_dim)(x)

# Latent Distribution
z_mean = Dense(latent_dim)(encoded)
z_log_var = Dense(latent_dim)(encoded)

# Sampling dari distribusi laten
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# Decoder
x = z
for units in hidden_layers[::-1]:
    x = Dense(units, activation='relu')(x)
decoded = Dense(input_dim, activation='sigmoid')(x)

# Membangun model DGMM
encoder = Model(input_img, z_mean)
decoder = Model(input_img, decoded)
autoencoder = Model(input_img, decoder(encoder(input_img)))

# Komponen Mixture Model
component_logits = Dense(num_components)(encoded)
component_probs = tf.nn.softmax(component_logits)
component_means = Dense(num_components * latent_dim)(encoded)
component_vars = Dense(num_components * latent_dim, activation=tf.nn.softplus)(encoded)

# Membangun model DGMM lengkap
dgmm = Model(input_img, [autoencoder.output, component_probs, component_means, component_vars])

# Menghitung loss function
def reconstruction_loss(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(K.binary_crossentropy(y_true, y_pred), axis=1))

def kl_divergence_loss(y_true, y_pred):
    kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1))
    return kl_loss

def mixture_loss(y_true, y_pred):
    num_samples = tf.shape(y_true)[0]
    y_true = tf.reshape(y_true, (num_samples, 1, input_dim))
    y_pred = tf.reshape(y_pred, (num_samples, num_components, input_dim))
    recon_loss = tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred + K.epsilon()), axis=-1))
    return recon_loss

def dgmm_loss(y_true, y_pred):
    recon_loss = reconstruction_loss(y_true, y_pred[0])
    mixture_loss_val = mixture_loss(y_true, y_pred[1])
    kl_loss = kl_divergence_loss(y_true, y_pred[2:])
    return recon_loss + mixture_loss_val + kl_loss

# Compile dan melatih DGMM
dgmm.compile(optimizer='adam', loss=dgmm_loss)

# Memuat dan mempersiapkan data fMRI
# Misalnya, x_train dan x_test adalah data fMRI yang telah dinormalisasi dan direshape sesuai dengan dimensi yang tepat.

# Melatih DGMM
dgmm.fit(x_train, [x_train, x_train, x_train, x_train],
         epochs=10,
         batch_size=32,
         shuffle=True,
         validation_data=(x_test, [x_test, x_test, x_test, x_test]))

# Menggunakan Encoder untuk mendapatkan representasi ruang laten
latent_representation = encoder.predict(x_test)

# Contoh penggunaan ruang laten
sample_latent_vector = latent_representation[0]
reconstructed_data = decoder.predict(np.array([sample_latent_vector]))

# Menampilkan hasil rekonstruksi
# Misalnya, Anda dapat melakukan analisis atau visualisasi pada reconstructed_data.

