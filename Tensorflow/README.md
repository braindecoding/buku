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

### Fungsi Layer Lambda

Layer `Lambda` di TensorFlow dan Keras digunakan untuk melibatkan operasi arbitrer atau fungsi sebagai layer dalam model Anda. `Lambda` layer bisa digunakan untuk menerapkan fungsi atau operasi sederhana yang tidak memerlukan bobot baru dan bisa didefinisikan dengan fungsi anonim atau lambda di Python.

Berikut adalah beberapa contoh penggunaan `Lambda` layer:

1. **Melakukan operasi matematika sederhana**: Misalnya, jika Anda ingin mengkuadratkan semua nilai dalam tensor, Anda dapat menggunakan `Lambda` layer.

    ```python
    from tensorflow.keras.layers import Lambda

    # Membuat layer Lambda yang akan mengkuadratkan inputnya
    square_layer = Lambda(lambda x: x ** 2)

    # Anda bisa menggunakan layer ini seperti layer lainnya
    output = square_layer(input_tensor)
    ```

2. **Mengubah bentuk tensor**: Kadang, Anda perlu mengubah bentuk tensor (misalnya, menggulung atau meratakan tensor). Anda dapat melakukan ini dengan `Lambda` layer.

    ```python
    from tensorflow.keras.layers import Lambda
    import tensorflow.keras.backend as K

    # Membuat layer Lambda yang akan menggulung inputnya
    reshape_layer = Lambda(lambda x: K.reshape(x, (-1, num_rows * num_cols)))

    # Anda bisa menggunakan layer ini seperti layer lainnya
    output = reshape_layer(input_tensor)
    ```

3. **Menerapkan fungsi nonlinier**: Misalnya, jika Anda ingin menerapkan fungsi nonlinier tertentu yang tidak disediakan oleh Keras atau TensorFlow, Anda dapat melakukannya dengan `Lambda` layer.

    ```python
    from tensorflow.keras.layers import Lambda
    import tensorflow as tf

    # Membuat layer Lambda yang akan menerapkan fungsi nonlinier
    nonlin_layer = Lambda(lambda x: tf.sin(x))

    # Anda bisa menggunakan layer ini seperti layer lainnya
    output = nonlin_layer(input_tensor)
    ```

4. **Menerapkan operasi kompleks**: Seperti yang kita lihat di contoh Variational Autoencoder sebelumnya, kita bisa menggunakan layer Lambda untuk melakukan operasi yang lebih kompleks, seperti sampling dari distribusi yang dipelajari.

    ```python
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    # Gunakan layer Lambda untuk melakukan sampling
    z = Lambda(sampling)([z_mean, z_log_var])
    ```

Ingatlah bahwa layer Lambda tidak memiliki bobot, jadi mereka tidak dapat belajar dari gradient selama proses pelatihan. Jadi, mereka hanya harus digunakan untuk operasi yang tidak memerlukan pembelajaran.

## Fungsi Model Compile
`model.compile()` adalah fungsi di TensorFlow dan Keras yang digunakan untuk mengkonfigurasi proses belajar (learning process) sebelum pelatihan model. Fungsi ini menerima tiga argumen penting:

1. **Optimizer**: Ini adalah algoritma yang digunakan untuk mengubah atribut dari model seperti bobot dan learning rate untuk mengurangi loss. Beberapa contoh optimizer termasuk SGD (Stochastic Gradient Descent), RMSprop, Adam, dan lainnya. Optimizer biasanya diteruskan sebagai string (nama dari optimizer) atau instance dari kelas optimizer.

2. **Loss function**: Ini adalah fungsi yang model coba minimalkan. Anda bisa merujuknya sebagai 'tujuan' yang model coba capai. Beberapa contoh loss function termasuk `mean_squared_error`, `categorical_crossentropy`, `binary_crossentropy`, dan lainnya.

### Custom Loss Function
Loss function adalah metrik yang digunakan untuk mengukur seberapa baik model belajar dari data. Pada beberapa kasus, loss function standar yang disediakan oleh Keras atau TensorFlow mungkin tidak cukup untuk menggambarkan tujuan optimasi yang tepat untuk model Anda. Dalam hal ini, Anda mungkin perlu mendefinisikan custom loss function.

Berikut beberapa alasan mengapa Anda mungkin perlu membuat custom loss function:

1. **Tujuan optimasi yang spesifik**: Misalnya, jika Anda bekerja pada suatu tugas yang perlu meminimalkan jenis kesalahan tertentu (seperti False Positives atau False Negatives) lebih dari jenis kesalahan lainnya, Anda mungkin ingin mendefinisikan loss function yang mencerminkan ini.

2. **Ketergantungan kompleks pada fitur atau label**: Kadang, Anda mungkin ingin menentukan loss function yang mempertimbangkan interaksi yang kompleks atau nonlinier antara fitur atau antara fitur dan label. 

3. **Regularisasi**: Anda mungkin ingin menambahkan bentuk regularisasi khusus ke loss function Anda yang tidak disediakan oleh Keras atau TensorFlow.

Berikut adalah contoh bagaimana mendefinisikan custom loss function di TensorFlow:

```python
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

model.compile(optimizer='adam', loss=custom_loss)
```

Dalam contoh ini, `custom_loss` adalah fungsi yang mengambil label sebenarnya (`y_true`) dan prediksi dari model (`y_pred`) dan mengembalikan rata-rata dari kuadrat perbedaannya. Model akan mencoba meminimalkan nilai ini selama pelatihan. Jadi, dalam hal ini, `custom_loss` sama dengan mean squared error (MSE), tetapi Anda bisa mengganti rumus di dalamnya dengan apa pun yang sesuai dengan kebutuhan Anda.

3. **Metrics**: Metrics digunakan untuk memantau kinerja model. Berbeda dengan loss function, metrics tidak digunakan saat pelatihan model tetapi digunakan untuk mengevaluasi kinerja model. Beberapa contoh metrics termasuk `accuracy`, `precision`, `recall`, dan lainnya.

Berikut adalah contoh penggunaan `model.compile()`:

```python
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
```

Pada contoh di atas, kita menggunakan Adam sebagai optimizer, sparse categorical crossentropy sebagai loss function, dan accuracy sebagai metric.