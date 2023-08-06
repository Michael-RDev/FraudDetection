from typing import Any
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation, Bidirectional
from keras.callbacks import EarlyStopping
from readData import x_train, y_train, x_val, y_val, max_length, max_words


def FraudModel(input_shape:tuple):
    model = Sequential()
    model.add(Embedding(input_dim=input_shape, output_dim=100, input_length=max_length))
    model.add(Bidirectional(LSTM(200, dropout=0.2, recurrent_dropout=0.2, recurrent_activation='sigmoid')))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(120, activation='sigmoid'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

earlyStawppp = EarlyStopping(patience=3, restore_best_weights=True)

model = FraudModel(input_shape=max_words)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

EPOCHS = 20
BATCH_SIZE = 32


model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_val, y_val), validation_split=0.2, batch_size=BATCH_SIZE, callbacks=earlyStawppp)

model.save("F:/FraudDetection/model/model.h5") #makes it easier to use predict