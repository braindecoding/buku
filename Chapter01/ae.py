# In[]: import
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

# In[]:Membangun Autoencoder
input_dim = 784  # Jumlah piksel dalam gambar MNIST
latent_dim = 64  # Jumlah dimensi dalam ruang laten

# In[]:Encoder
input_img = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(latent_dim, activation='relu')(encoded)

# In[]:Decoder
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# In[]:Membangun model Autoencoder
autoencoder = Model(input_img, decoded)

# In[]:Membangun model Encoder terpisah
encoder = Model(input_img, encoded)
print("============================ encoder : =============================")
encoder.summary()
# In[]:Compile dan melatih Autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
print("============================ Autoencoder : =============================")
autoencoder.summary()
# In[]:Memuat dan mempersiapkan data MNIST
from keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

# In[]:Normalisasi dan flatten data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# In[]:Melatih Autoencoder
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# In[]:Menggunakan Encoder untuk mendapatkan representasi ruang laten ukuran 64
latent_representation = encoder.predict(x_test)

# In[]:Contoh penggunaan ruang laten
sample_latent_vector = latent_representation[0]
reconstructed_image = autoencoder.predict(np.array([sample_latent_vector]))

# In[]:Menampilkan hasil rekonstruksi
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
