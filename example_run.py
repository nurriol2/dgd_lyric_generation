#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import requests
import numpy as np

#model = load_model("/Users/nikourriola/Desktop/projects/dgd_lyric_generation/dgd_lyric_gen.h5", compile=False)

#a single .csv containing Dance Gavin Dance lyrics 
filepath = "https://raw.githubusercontent.com/nurriol2/dgd_lyric_generation/ft-rnn/dance_gavin_dance_lyrics.txt"
text = requests.get(filepath).text
#normalization step by reducing the vocabulary size
text = text.lower()
vocab = sorted(set(text))
#encoding characters as integers
char2idx = {u:i for i, u in enumerate(vocab)}
#a decode map to get text as output, instead of integers
idx2char = np.array(vocab)

def generate_text(model, start_string):
    """
    Generate text using a trained model
    
    Args:
    model (tensorflow.keras.Model):  A trained model
    start_string (str):  The starting input sequence
    
    Returns:
    (str):  The predicted text. Concatenation of start_string and following `num_generate` predicted characters.
    """

    #number of characters to generate
    num_generate = 1000

    #converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    #storing the predicted indices (wrt look-up table)
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 0.97

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"[intro: "))