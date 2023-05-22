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

# Deep Generative Mixture Model (DGMM) 

# DGMM dengan fmri dan stimulus