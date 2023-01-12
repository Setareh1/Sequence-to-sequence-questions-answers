#This script used for inference
#To perform inference, we feed the new input seq to the encoder
#Grab the internal state of the encoder
#Iterate on the decoder:
#   pass the internal state of the encoder to decoder
#   starting with the "[START]" symbol, perform next word prediction to get the
#   first predicted word.
#   append the first word to the "[START]" and then use that as the input to 
#   the decoder
#   repeat until we generate [STOP] symbol

# Don't forget to
# export CUDA_VISIBLE_DEVICES=0
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"


#######################################################################


import numpy as np

import tensorflow as tf
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import shelve, re


#######################################################################


## This code is needed to run on Mist, the SciNet GPU cluster.  In
## most situations it can probably be removed.
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()


#######################################################################


input_text = "It's hot out today."

shelvefile = 'qa.S2S.metadata.shelve'
modelfile = 'qa.S2S.model.h5'


#######################################################################


## Get the metadata.
print('Reading metadata shelve file.')
g = shelve.open('data/' + shelvefile, flag = 'r')

tokenizer = g['tokenizer']
latent_size = g['latent_size']
lstm_size = g['lstm_size']
input_size = g['input_size']
max_sentence_length = g['max_sentence_length']
VOCAB_SIZE = g['VOCAB_SIZE']
 
g.close()


#######################################################################


## The function used to clean the data.  Needed to clean the input
## sentence.
def clean_text(text):
    ## Let's try this other guy's approach.
    sentence = text.lower().strip()
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    return sentence.strip()


#######################################################################

## For inference we build two networks out of the original network,
## one for the encoder and one for the decoder.

## First, get the trained model.
model = km.load_model('data/' + modelfile)

## Create a dictionary containing all layer (references each layer by name).
layers = {layer.name: layer for layer in model.layers}

#build encoding layer
encoder_input = kl.Input(shape = (None,))

encoder_embedding = layers["encoder_embedding"]
encoder_embedding_output = encoder_embedding(encoder_input)

encoder_lstm = layers["encoder_lstm"]
_, encoder_state_h, encoder_state_c = \
                    encoder_lstm(encoder_embedding_output)

# We discard encoder output and only keep the states to pass the decoder.
encoder_states = [encoder_state_h, encoder_state_c]

## Build the separate encoder model.
encoder_model = km.Model(encoder_input, encoder_states)


## And now the decoder.
decoder_input = kl.Input(shape = (None,))

decoder_embedding = layers["decoder_embedding"]
decoder_embedding_output = decoder_embedding(decoder_input)

## We build allow ourselves the ability to set the initial state of
## the decoder by building a separate input layer to do so.
decoder_state_input_h = kl.Input(shape=(None,))
decoder_state_input_c = kl.Input(shape=(None,))
decoder_state_input = [decoder_state_input_h,
                       decoder_state_input_c]

decoder_lstm = layers["decoder_lstm"]
decoder_lstm_output = decoder_lstm(decoder_embedding_output,
                                   initial_state = decoder_state_input)

decoder_dense = layers["output"]
decoder_output = decoder_dense(decoder_lstm_output)

## The final decoder model.
decoder_model = km.Model([decoder_input] + decoder_state_input,
                         decoder_output)

## You might be wondering about the plus sign in the above model
## declaration.  See this link for an explanation:
## https://stackoverflow.com/questions/49937629/keras-model-input-syntax-use-of-plus

#######################################################################

## Set the start and end tokens.
START_TOKEN = [tokenizer.vocab_size]
END_TOKEN = [tokenizer.vocab_size + 1]

## Tokenize the input data.
tokenized_input = START_TOKEN + \
                  tokenizer.encode(clean_text(input_text)) + \
                  END_TOKEN

## Create an empty array to hold the encoder input, and fill it.
## This is just an array but passing through keras the batch size should
## be specified (1,max_len)
encoder_input_data = np.zeros((1, max_sentence_length), dtype = np.int)
encoder_input_data[0,:len(tokenized_input)] = tokenized_input

#print('encoder data:', encoder_input_data)

## Get the output states of the encoder, based on the input data.
encoder_states_value = encoder_model.predict(encoder_input_data)

## Create an empty input array for the decoder, and make the first
## entry the START_TOKEN.
decoder_input_data = np.zeros((1, max_sentence_length), dtype = np.int)
decoder_input_data[0, 0] = START_TOKEN[0]


done = False
decoded_sentence = ''
i = 1


## Now loop until we get and END_TOKEN out of the decoder.
while not done:

    ## Input the current sentence, plus the state of the decoder.
    ## Initially this is the START_TOKEN and the encoder's final
    ## internal state.  As we continue looping the input sentence to
    ## the decoder will grow, as we add words, but the initial
    ## internal state of the decoder will remain the final internal
    ## state of the encoder.
    output_tokens = decoder_model.predict(
        [decoder_input_data] + encoder_states_value)


    ## Get the index of the next predicted word, from the decoder.
    sampled_token_index = np.argmax(output_tokens[0, i - 1, :])

    ## If we're not done, then get the word and add it to the text.
    if sampled_token_index != END_TOKEN[0]:

        word = tokenizer.decode([sampled_token_index])
        decoded_sentence += word


    ## If we are done, or the output is getting too long, then stop.
    if ((sampled_token_index == END_TOKEN[0]) or
        (len(decoded_sentence.split(' ')) > max_sentence_length)):
        done = True

    ## Add the new word to the running sentence, for the next loop
    ## iteration.
    decoder_input_data[0, i] = sampled_token_index

    i += 1
    

## Print out the result.
print()
print("input sentence")
print(input_text)
print()
print("output sentence")
print(decoded_sentence)
