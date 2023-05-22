# Variational Auto Encoder

VAE atau kepanjangan dari variational auto encoder beberapa arsitekturnbya

# Lambda
Fungsi Lambda adalah salah satu lapisan yang disediakan oleh framework Keras. Lapisan Lambda digunakan untuk membuat lapisan khusus yang melakukan operasi kustom pada tensor input.

Dalam konteks Variational Autoencoder (VAE) pada contoh sebelumnya, lapisan Lambda digunakan untuk melakukan sampling dari distribusi laten normal. Operasi sampling ini tidak dapat diwakili oleh lapisan bawaan yang disediakan oleh Keras, sehingga kita menggunakan lapisan Lambda untuk membuat lapisan khusus yang dapat melakukan operasi tersebut.

Contoh penggunaan Lambda dalam contoh VAE adalah sebagai berikut:

```py
from keras.layers import Lambda

# Latent Distribution
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])
```

fungsi loss dalam VAE
```py
# Loss function VAE
reconstruction_loss = mean_squared_error(K.flatten(input_fMRI), K.flatten(decoded))
reconstruction_loss *= input_dim
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)
```

membangun VAE Model
```py
# Build VAE model
encoder = Model(input_img, z_mean)
decoder = Model(decoder_input, decoded)
vae = Model(input_img, decoder(z))
```
untuk CVAE
```py
# Build CVAE model
encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
decoder = Model(decoder_inputs, decoder_outputs, name='decoder')
cvae_outputs = decoder(encoder(encoder_inputs)[2])
cvae = Model(encoder_inputs, cvae_outputs, name='cvae')
```

# Deep Generative Mixture Model (DGMM) 

# DGMM dengan fmri dan stimulus

# Menggabungkan dua input gambar dan fmri

```py
input_shape = (width, height, channels)  # Dimensi gambar
input_fMRI = Input(shape=(input_dim,))  # Input fMRI
x = Reshape(input_shape)(input_fMRI)  # Reshape fMRI menjadi gambar dengan dimensi yang sesuai
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
output = Dense(output_dim, activation='softmax')(x)  # Output gambar
```