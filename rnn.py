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
seq_length = 100 #NB tunable parameter
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

#entire dataset split into batches of seq_length
dataset = sequences.map(split_input_target)

#text is considered time series data because each character is interpreted as a single timestep
#for each timestep, the RNN will try to predict the next timestep
#an example of desired outcome is shown here
"""
for input_example, target_example in  dataset.take(1):
    for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
        print("Step {:4d}".format(i))
        print("  input: {} ({:s})".format(input_idx, repr(idx_to_characters[input_idx])))
        print("  expected output: {} ({:s})".format(target_idx, repr(idx_to_characters[target_idx])))
"""
#create training batches
BATCH_SIZE = 64
#NB tf shuffles the data inside a buffer; that way can handle (possibly) inf. sequences
BUFFER_SIZE = 10000
#shuffle data
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

#RNN model building

#QUESTION: is this explanation correct?
#the size of the vector characters are mapped to
embedding_dim = 256
#the output is an index so the final layer needs to accomodate the vocabulary size
vocab_size = len(vocab)
#number of rnn units
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    #see docs for why `stateful=True`
    model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
            tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),
            tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(vocab_size=vocab_size, 
                    embedding_dim=embedding_dim, 
                    rnn_units=rnn_units, 
                    batch_size=BATCH_SIZE)

#check that the input is the correct shape
"""
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")


    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
    print("Input: \n", repr("".join(idx_to_characters[input_example_batch[0]])))
    print()
    print("Next Char Predictions: \n", repr("".join(idx_to_characters[sampled_indices ])))
"""
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)
# directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 30

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback]).history

tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

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

predicted_text = generate_text(model, start_string=u"[Intro: ")
with open("predicted_text_16_08_20.txt", 'wb') as f:
    f.write(predicted_text)