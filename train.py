import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json

with open('dataset.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

dataset = data

texts = [message["text"] for message in dataset]
labels = [message["state"] for message in dataset]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts)

max_sequence_len = max([len(sequence) for sequence in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='post')

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word_index) + 1, 16, input_length=max_sequence_len),
    tf.keras.layers.GRU(16),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
    # we think, GRU is better than this
    # tf.keras.layers.Embedding(len(word_index) + 1, 16, input_length=max_sequence_len),
    # tf.keras.layers.GlobalAveragePooling1D(),
    # tf.keras.layers.Dense(16, activation='relu'),
    # tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.003), 
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

model.fit(padded_sequences, np.array(labels), epochs=200, verbose=1)#, batch_size = 32)

model.save('./pretrained/model')

tokenizer_json = tokenizer.to_json()
with open('./pretrained/tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

max_sequence_len_file = open("./pretrained/max_sequence_len.txt", "w", encoding="utf-8")
max_sequence_len_file.write(str(max_sequence_len))
max_sequence_len_file.close()
