#!/usr/bin/env python3

import flask
from flask import Flask, render_template
from keras.models import load_model
import tensorflow as tf

#use keras to load the model
model = load_model("model/dgd_lyric_gen.h5", compile=False)

#character mappings from Dance Gavin Dance lyrics dataset
char2idx = {'\n': 0, ' ': 1, '!': 2, '"': 3, '&': 4, "'": 5, '(': 6, ')': 7, '+': 8, ',': 9, '-': 10, '.': 11, '0': 12, '1': 13, '2': 14, '3': 15, '4': 16, '5': 17, '6': 18, '7': 19, '8': 20, '9': 21, ':': 22, ';': 23, '?': 24, '[': 25, ']': 26, 'a': 27, 'b': 28, 'c': 29, 'd': 30, 'e': 31, 'f': 32, 'g': 33, 'h': 34, 'i': 35, 'j': 36, 'k': 37, 'l': 38, 'm': 39, 'n': 40, 'o': 41, 'p': 42, 'q': 43, 'r': 44, 's': 45, 't': 46, 'u': 47, 'v': 48, 'w': 49, 'x': 50, 'y': 51, 'z': 52, 'ç': 53, 'é': 54, 'í': 55, 'ú': 56, '\u2005': 57, '\u200a': 58, '’': 59, '‚': 60, '“': 61, '\u205f': 62}
idx2char = ['\n', ' ', '!', '"', '&', "'", '(', ')', '+', ',', '-', '.', '0',
       '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', '[',
       ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
       'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
       'z', 'ç', 'é', 'í', 'ú', '\u2005', '\u200a', '’', '‚', '“',
       '\u205f']

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

    output_sequence = start_string + ''.join(text_generated)
    return output_sequence

def foobar(seq):
    return seq+"PINEAPPLES"

app = Flask(__name__, template_folder="templates")

@app.route('/', methods=["GET", "POST"])
def main():
    if flask.request.method=="GET":
        return render_template("main.html")

    
    if flask.request.method=="POST":
        seq_in = flask.request.form["input_sequence"]
        seq_in = seq_in.lower()
        prediction = generate_text(model, seq_in)
        return flask.render_template("main.html", original_input={"input_sequence":seq_in}, result=prediction)
    """
    if flask.request.method=="POST":
        seq_in = flask.request.form["input_sequence"]
        seq_in = seq_in.lower()
        prediction = foobar(seq_in)
    return flask.render_template("main.html", original_input={"input_sequence":seq_in}, result=prediction)
    """
#added so app can be run using terminal like normal module
if __name__=="__main__":
    app.run()