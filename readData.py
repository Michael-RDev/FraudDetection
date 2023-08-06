import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

dataFile = pd.read_csv("F:/FraudDetection/data/email_spam.csv")

email_text = dataFile['text'].values

email_type = dataFile['type'].values


max_words = len(set(email_text))
tokenizer = Tokenizer(num_words=max_words) 
tokenizer.fit_on_texts(email_text)
tokenized_words = tokenizer.texts_to_sequences(email_text)

max_length = max([len(sequences) for sequences in tokenized_words])
x_data = pad_sequences(tokenized_words, maxlen=max_length)


y_data = np.array([1 if label == 'spam' else 0 for label in email_type])

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
