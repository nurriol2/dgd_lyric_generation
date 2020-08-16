#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import time

text = open("dance_gavin_dance_lyrics.txt", "rb").read().decode(encoding='utf-8')

#set a unique vocabulary of characters
vocab = sorted(set(text)) #89 unique characters

#vectorize the text
characters_to_idx = {char:idx for idx, char in enumerate(vocab)}
idx_to_characters = np.array(vocab)

#encode the entire text 
text_as_int = np.array([characters_to_idx[char] for char in text])

#create training examples and targets

#maximum length sentence we want for a single input in characters
seq_length = 100 #QUESTION:  arbitrarily chosen?
examples_per_epoch = len(text)//(seq_length+1)

#stream of individual character indices
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

#convert indiidual character indices to sequences w/ specific size
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

#for a given input, model should predict THE NEXT CHARACTER
#.:. a target is the input sequence shifted once to the right
def split_input_target(chunk):
    """Make training data and associated target from a fixed length sequence of indices

    Args:
        chunk (tensorflow.tensor): Sequence of indices

    Returns:
        tuple: The training example and associated target as a tuple pair
    """
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx_to_characters[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx_to_characters[target_example.numpy()])))
   

  


