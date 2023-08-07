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