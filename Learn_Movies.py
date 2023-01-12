#This script trains an sequence-to-sequence network to predict the
# next word in a text file.  In this case it is trained on the Cornell
# Movie Dialog data set.
#


# Don't forget to
import os
os.environ['CUDA_VISIBLE_DEVICES']="3"


#######################################################################


import numpy as np
import os
import pickle
import glob
import re

import tensorflow as tf
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import shelve

import tensorflow_datasets as tfds
import tensorflow.keras.preprocessing as tfkp


#######################################################################


## This code is needed to run on Mist, the SciNet GPU cluster.  In
## most situations it can probably be removed.
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

    
#######################################################################


# Specify some data file names.
inputpath = os.path.expanduser('./data/') #os.path.expanduser('~/work/data/Movie.Dialogs/cornell.movie-dialogs.corpus')
#inputpath = os.path.expanduser('.')
inputfile = 'questions_answers.npy'
datafile = 'qa'
shelvefile = 'qa.S2S.metadata.shelve'
modelfile = 'qa.S2S.model.h5'

max_sentence_length = 40

## Size of the Encoding layer and the LSTM layers.
latent_size = 64 #embedding layer of output size of 64
lstm_size = 64  #64 memory cells in LSTMs

num_epochs = 250
batch_size = 128


#######################################################################

#*******************************************
# PREPROCESS THE DATA: convert the data into 2 lists of cleaned ques and ans!!!
#*******************************************


def clean_text(text):

    ## This is much more efficient that my previous approach.
    sentence = text.lower().strip()
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    return sentence.strip()


#######################################################################


# If the data have already been processed, then don't do it again,
# just read it in.
if (not os.path.isfile('data/' + shelvefile + '.dat')):

    # Read in the entire file.
    print('Reading data.')

    encoder_input0 = []
    decoder_input0 = []

    data = np.load(inputpath + '/' + inputfile)
    
    questions = data[:,0]
    answers = data[:,1]

    for question, answer in zip(questions, answers):
        print(question)
        q = clean_text(question.decode('UTF-8'))
        a = clean_text(answer.decode('UTF-8'))
        print(q)
        encoder_input0.append(q)
        
        decoder_input0.append(a)
            

    print("Total number of q-a pairs is", len(encoder_input0))

    ## We now train a tokenizer on the data.
    # tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    #     encoder_input0 + decoder_input0, target_vocab_size = 2**13)

    # **************************************************************
    #the encoder input data set, consisting of the input data
    #the decoder input data set, consisting of the target data, used as inputs
    #for next-word-prediction training of the decoder

    # We tokenize both input and target data (encoder (questions)+decoder (ans))
    # The output if the decoder is treated as a cetegorization problem 
    #We will do one-hot encoding the vocab for target of decoder(next word in seq)
    # and its a standard categorization problem


    #tfds.deprecated.text --> we do subword tokenization on inputs. We use
    #one of the built-in tokenizer that comes from tensor flow data sets and train 
    #it on the data
    #The tokenizer will analyze all of the text that is in there
    #It will determine what the best particular set of tokens is
    #It then creates vocabulary
    #And then we have tokenizer model to apply on the any text
    # **************************************************************

    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        encoder_input0 + decoder_input0, target_vocab_size = 2**13)




    ## Add start and end tokens to the vocabulary.
    # Give the start and end token indices
    START_TOKEN = [tokenizer.vocab_size]
    END_TOKEN = [tokenizer.vocab_size + 1]

    VOCAB_SIZE = tokenizer.vocab_size + 2

    print("Vocab size is ", VOCAB_SIZE)

    #*****
    #Now we make lists of takenized questions and answers
    tokenized_questions, tokenized_answers = [], []
    ##****
    # Loop over the lists of questions and answers and build lists of
    #tokenized q and a
    for (sentence1, sentence2) in zip(encoder_input0, decoder_input0):

        ## tokenize sentence                                  
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

        ## check tokenized sentence max length                                
        if (len(sentence1) <= max_sentence_length and
            len(sentence2) <= max_sentence_length):
            tokenized_questions.append(sentence1)
            tokenized_answers.append(sentence2)

            
    print('Encoding data.')

    ## pad tokenized sentences, so they're all the same length.
    encoder_input_data = tfkp.sequence.pad_sequences(
        tokenized_questions, maxlen = max_sentence_length,
        padding = 'post')
    decoder_input_data = tfkp.sequence.pad_sequences(
        tokenized_answers, maxlen = max_sentence_length,
        padding = 'post')

            
    input_size = len(tokenized_questions)

    ## Initialize empty arrays to hold the decoder target.
    ##*******
    #Remember the target for the decoder is one-hot-encoded targets for each
    #token in the answers, input size of matrix max_length by vocab size
    decoder_target_data = np.zeros([input_size, max_sentence_length,
                                    VOCAB_SIZE], dtype = np.bool)


    for i in range(input_size):

        ## The decoder target data is offset by 1, since the target of
        ## the decoder is one after the input of the decoder.
        for j in range(1, max_sentence_length):
            decoder_target_data[i, j - 1,
                                decoder_input_data[i, j]] = 1

                
    # The processing of the data takes a fair amount of time.  Save
    # the data so we don't have to do this again.  We do this in a
    # numpy file since the data is large and the shelve can't handle
    # it.
    print('Saving processed data.')
    np.save('data/' + datafile + '.encoder.input.npy', encoder_input_data)
    np.save('data/' + datafile + '.decoder.input.npy', decoder_input_data)
    np.save('data/' + datafile + '.decoder.target.npy', decoder_target_data)

    # Do the same with the metadata.
    print('Creating metadata shelve file.')
    g = shelve.open('data/' + shelvefile)
    g['tokenizer'] = tokenizer
#    g['max_encoder_length'] = max_encoder_length
    g['max_sentence_length'] = max_sentence_length
    g['input_size'] = input_size
    g['latent_size'] = latent_size
    g['lstm_size'] = lstm_size
    g['VOCAB_SIZE'] = VOCAB_SIZE
    g.close()

else:

    # If the data already exists, then use it.
    
    print('Reading metadata shelve file.')
    g = shelve.open('data/' + shelvefile, flag = 'r')

    print(list(g.keys()))

    tokenizer = g['tokenizer']
    latent_size = g['latent_size']
    lstm_size = g['lstm_size']
    input_size = g['input_size']
    max_sentence_length = g['max_sentence_length']
    VOCAB_SIZE = g['VOCAB_SIZE']
    g.close()

    print('Reading processed data.')
    encoder_input_data = np.load('data/' + datafile +
                                 '.encoder.input.npy')
    decoder_input_data = np.load('data/' + datafile +
                                 '.decoder.input.npy')
    decoder_target_data = np.load('data/' + datafile +
                                  '.decoder.target.npy')


# If this is our first rodeo, build the model.
if (not os.path.isfile('data/' + modelfile)):

    print('Building network.')

    ## The encoder input layer.
    encoder_input = kl.Input(shape = (None,), name = 'encoder_input')

    ## The encoder embedding layer.  Note that it takes an input
    ## length, so that it can return a sequence.  As such the input is
    ## of size (batch_size, max_sentence_length) and the output of the
    ## layer is (batch_size, max_sentence_length, latent_size).  Each
    ## 'word' of the input sequence is run through the embedding layer
    ## separately.
    ##******
    #The matrix of the values that are gonna be trained has vocab size
    # as the num of rows and latent size as the output. Each row is the index of
    #word from vocabulary and the output is a row with size of latent size(64)

    encoder_embedding = kl.Embedding(VOCAB_SIZE, latent_size,
                                     name = "encoder_embedding",
                                     input_length = max_sentence_length)
    # apply the embedding layer onto the encoder input                                 
    encoder_embedding_output = encoder_embedding(encoder_input)
    #Next we build LSTM
    ## Encoder LSTM.  Note that we explicitly tell the layer to output
    ## the hidden state of the layer.  These are output as the 'hidden
    ## state' and the 'cell state'.  These are actually the last
    ## output value of the LSTM, and the internal state of the memory
    ## cells. Because we want to pass it to decoder to initialize the decode
    ## internal state
    encoder_lstm = kl.LSTM(lstm_size, return_state = True,
                           name = "encoder_lstm",
                           input_shape = (max_sentence_length, latent_size))
    ## First entry is the output that is discarded
    _, state_h, state_c = encoder_lstm(encoder_embedding_output)

    # We discard the encoder output and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder input layer.
    decoder_input = kl.Input(shape = (None,), name = 'decoder_input')

    ## The decoder embedding layer, same as the encoding embedding
    ## layer.
    decoder_embedding = kl.Embedding(VOCAB_SIZE, latent_size,
                                     name = "decoder_embedding",
                                     input_length = max_sentence_length)
    decoder_embedding_output = decoder_embedding(decoder_input)


    # We set up our decoder to return full output sequences, not just
    # a single output.  We will use the full sequence when we do
    # inference using the decoder.  Note that the initial state of the
    # decoder is set to the encoder states.

    #return_sequences = True --> means that the decoder is going to return
    #the output of LSTM (hidden state) for every entry in the input seq not #
    # that the last entry. We build a fully-NN on these to predict word
    #But for entire sequence that is input to LSTM. We need the sequence per entry 
    #as we need to take 
    #that output and add a new word on it and pass it as input for next iteration
    #We initialize the initial_state for decoder LSTM to be encoder_states
    decoder_lstm = kl.LSTM(lstm_size, return_sequences = True,
                           name = "decoder_lstm",
                           input_shape = (max_sentence_length,latent_size))
    decoder_lstm_output = decoder_lstm(decoder_embedding_output,
                                        initial_state = encoder_states)

    ## The decoder output layer. 
    # ##The fully connected layer applied on decoder output

    decoder_dense = kl.Dense(VOCAB_SIZE, activation = 'softmax', 
                             name = 'output')
    decoder_output = decoder_dense(decoder_lstm_output)

    ## The final model.
    model = km.Model([encoder_input, decoder_input], decoder_output)

    print(model.summary())

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam',  # rmsprop?
                  metrics = ['accuracy'])

else:

    # Otherwise, use the previously-saved model as our starting point
    # so that we can continue to improve it.
    
    print('Reading model file.')
    model = km.load_model('data/' + modelfile)


# Fit!  Begin elevator music...
print('Beginning fit.')
fit = model.fit([encoder_input_data, decoder_input_data],
                decoder_target_data, epochs = num_epochs,
                batch_size = batch_size, verbose = 2,
                validation_split = 0.1)

# Save the model so that we can use it as a starting point.
model.save('data/' + modelfile)
