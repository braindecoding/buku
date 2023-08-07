# Pengenalan Tensorflow

## Pembuatan Model
Setelah Anda mendeklarasikan layer-layer yang diinginkan, Anda perlu menambahkannya ke dalam sebuah model sebelum dapat melatihnya. Ada beberapa cara untuk melakukannya di TensorFlow menggunakan Keras, dua yang paling populer adalah dengan `Sequential` API dan `Functional` API.

1. **Sequential API**

   Dengan API ini, Anda dapat dengan mudah menambahkan layer ke dalam model Anda satu per satu dalam urutan dari input ke output. Berikut adalah contoh:

   ```python
   from tensorflow.keras import Sequential
   from tensorflow.keras.layers import Dense

   # Membuat model
   model = Sequential()

   # Menambahkan layer pertama
   model.add(Dense(units=64, activation='relu'))

   # Menambahkan layer kedua
   model.add(Dense(units=32, activation='relu'))

   # Menambahkan layer output
   model.add(Dense(units=10, activation='softmax'))
   ```

   Dalam contoh ini, kita membuat sebuah model dengan dua layer tersembunyi (keduanya menggunakan fungsi aktivasi ReLU) dan sebuah layer output dengan 10 unit (menggunakan fungsi aktivasi softmax).

2. **Functional API**

   Functional API memberikan lebih banyak fleksibilitas dan memungkinkan Anda untuk membuat model yang memiliki topologi yang lebih kompleks, seperti model multi-input atau multi-output, model dengan layer bersama, dll. Berikut adalah contoh penggunaannya:

   ```python
   from tensorflow.keras import Model, Input
   from tensorflow.keras.layers import Dense

   # Membuat input layer
   inputs = Input(shape=(32,))

   # Menambahkan layer pertama
   x = Dense(units=64, activation='relu')(inputs)

   # Menambahkan layer kedua
   x = Dense(units=32, activation='relu')(x)

   # Menambahkan layer output
   outputs = Dense(units=10, activation='softmax')(x)

   # Membuat model
   model = Model(inputs=inputs, outputs=outputs)
   ```

   Dalam contoh ini, kita membuat model yang sama seperti sebelumnya, tetapi dengan menggunakan Functional API.

Setelah menambahkan layer ke model, Anda biasanya ingin mengkompilasi model (yaitu, menentukan loss function, optimizer, dan metrik) dan kemudian melatihnya menggunakan data Anda.

## model dengan layer bersama

### NLP

Layer bersama dalam deep learning merujuk kepada konsep di mana layer yang sama, dengan bobot yang sama, digunakan di beberapa tempat dalam model. Artinya, output dari layer ini akan bergantung pada beberapa set input yang berbeda, dan gradien dari layer ini akan diakumulasikan dari semua tempat di mana layer ini digunakan. Ini adalah teknik yang sering digunakan untuk model yang memiliki beberapa input yang harus diproses dengan cara yang sama.

Contoh paling umum penggunaan layer bersama adalah dalam model yang menganalisis dua teks untuk melihat seberapa mirip mereka. Dalam kasus ini, kedua teks tersebut akan diolah oleh layer yang sama, karena kita ingin setiap teks diproses dengan cara yang sama untuk memastikan bahwa perbandingan mereka adil.

Berikut adalah contoh bagaimana ini dapat diimplementasikan menggunakan API Fungsional Keras:

```python
from tensorflow.keras.layers import Dense, Input, Embedding, LSTM
from tensorflow.keras.models import Model

# Jumlah maksimum fitur teks kita
max_features = 10000
# Panjang maksimum teks kita
maxlen = 100

# Membuat layer Embedding yang akan digunakan secara bersama
shared_embedding = Embedding(max_features, 128)

# Membuat input pertama
input_a = Input(shape=(maxlen,), dtype='int32')

# Membuat input kedua
input_b = Input(shape=(maxlen,), dtype='int32')

# Kita menggunakan layer Embedding yang sama untuk mengolah kedua input
processed_a = shared_embedding(input_a)
processed_b = shared_embedding(input_b)

# Membuat layer LSTM untuk diproses lebih lanjut
lstm_layer = LSTM(32)

# Proses kedua input dengan LSTM
output_a = lstm_layer(processed_a)
output_b = lstm_layer(processed_b)

# Menggabungkan kedua output
merged_vector = tf.keras.layers.concatenate([output_a, output_b], axis=-1)

# Membuat layer Dense untuk prediksi akhir
predictions = Dense(1, activation='sigmoid')(merged_vector)

# Membuat model dengan dua input dan satu output
model = Model([input_a, input_b], predictions)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
```

Dalam contoh ini, kita membuat sebuah model yang mengambil dua teks sebagai input, memproses kedua teks tersebut dengan layer embedding dan LSTM yang sama, dan menggabungkan outputnya untuk membuat prediksi akhir.

### VAE

Variational Autoencoder (VAE) adalah salah satu tipe model di mana kita mungkin menemui penggunaan layer bersama. Dalam VAE, struktur encoder dan decoder seringkali simetris. Sehingga, biasanya layer dalam encoder dan decoder tidak benar-benar 'dibagi', tetapi memiliki struktur yang mirip. Namun, dalam beberapa kasus tertentu, Anda bisa menggunakan beberapa layer yang sama dalam encoder dan decoder.

VAE bekerja dengan cara mempelajari distribusi probabilitas dari data input dan mencoba untuk menghasilkan data baru dari distribusi ini. Berikut adalah contoh bagaimana VAE bisa dibuat menggunakan TensorFlow dan Keras:

```python
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K

# Dimensi input dan output
original_dim = 28 * 28
# Dimensi latent space (dimensi distribusi probabilitas yang akan dipelajari)
latent_dim = 2

# Membuat input layer
inputs = Input(shape=(original_dim,))

# Membuat layer Dense untuk encoder
encoder_layer = Dense(64, activation='relu')
encoder_output = encoder_layer(inputs)

# Membuat layer Dense untuk mu dan log_var
z_mean = Dense(latent_dim)(encoder_output)
z_log_var = Dense(latent_dim)(encoder_output)

# Fungsi untuk sampling dari distribusi yang kita pelajari
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Gunakan layer Lambda untuk melakukan sampling
z = Lambda(sampling)([z_mean, z_log_var])

# Membuat layer Dense untuk decoder
decoder_layer = Dense(64, activation='relu')
decoder_output = decoder_layer(z)

# Membuat layer output
outputs = Dense(original_dim, activation='sigmoid')(decoder_output)

# Membuat model
vae = Model(inputs, outputs)
```

Catatan: Contoh di atas adalah implementasi dasar dari VAE dan tidak termasuk penanganan loss yang spesifik yang biasanya diperlukan dalam VAE. Biasanya, loss dari VAE terdiri dari dua bagian: bagian pertama adalah perbedaan antara input dan output (reconstruction loss), dan bagian kedua adalah divergence Kullback-Leibler antara distribusi yang dipelajari dan distribusi normal standar. Anda perlu menambahkan bagian ini jika Anda ingin melatih VAE.