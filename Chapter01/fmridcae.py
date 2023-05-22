import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

# Membangun Autoencoder
input_dim = 1000  # Jumlah fitur dalam data fMRI
latent_dim = 64  # Jumlah dimensi dalam ruang laten
hidden_layers = [512, 256, 128]  # Jumlah unit dalam setiap lapisan tersembunyi

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
autoencoder.compile(optimizer='adam', loss='mse')

# Memuat dan mempersiapkan data fMRI
# Misalnya, x_train dan x_test adalah data fMRI yang telah dinormalisasi dan direshape sesuai dengan dimensi yang tepat.

# Melatih Autoencoder
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=32,
                shuffle=True,
                validation_data=(x_test, x_test))

# Menggunakan Encoder untuk mendapatkan representasi ruang laten
latent_representation = encoder.predict(x_test)

# Contoh penggunaan ruang laten
sample_latent_vector = latent_representation[0]
reconstructed_data = autoencoder.predict(np.array([sample_latent_vector]))

# Menampilkan hasil rekonstruksi
# Misalnya, Anda dapat melakukan analisis atau visualisasi pada reconstructed_data.

