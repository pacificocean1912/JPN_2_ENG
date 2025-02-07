import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
from icecream import ic
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import datasets
from datasets import load_dataset
import torch
from numpy import asarray, zeros
import cupy as cp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ic('Using device:', device)
if device.type == 'cuda':
    ic(torch.cuda.get_device_name(0))
    ic('Memory Usage:')
    ic('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    ic('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

ic("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

BATCH_SIZE = 64
EPOCHS = 20
LSTM_NODES = 256
NUM_SENTENCES = 200000
MAX_SENTENCE_LENGTH = 50
MAX_NUM_WORDS = 200000
EMBEDDING_SIZE = 200
SIZE_OF_BLOCK = 10000
embeddings_dictionary = dict()
try:
    with open(f'glove/glove.6B.{EMBEDDING_SIZE}d.txt', encoding='utf-8') as glove_file:
        for line in glove_file:
            rec = line.split()
            word = rec[0]
            vector_dimensions = asarray(rec[1:], dtype='float')
            embeddings_dictionary[word] = vector_dimensions
except FileNotFoundError:
    raise RuntimeError("GloVe file not found")

try:
    train_data = datasets.load_from_disk("dataset_jpn_eng.hf/train")
    test_data = datasets.load_from_disk("dataset_jpn_eng.hf/test")
    valid_data = datasets.load_from_disk("dataset_jpn_eng.hf/valid")
except Exception as e:
    raise RuntimeError("Error loading datasets: {}".format(e))

input_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)
input_tokenizer.fit_on_texts(train_data['eng'])
input_integer_seq = input_tokenizer.texts_to_sequences(train_data['eng'])

word2idx_inputs = input_tokenizer.word_index
ic('Total unique words in the input: %s' % len(word2idx_inputs))

max_input_len = max(len(sen) for sen in input_integer_seq)
ic("Length of longest sentence in input: %g" % max_input_len)

encoder_input_sequences = pad_sequences(input_integer_seq, maxlen=max_input_len)
ic("encoder_input_sequences.shape:", encoder_input_sequences.shape)

output_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS, filters='')
output_tokenizer.fit_on_texts(train_data['jpn'] + train_data['eng'])
output_integer_seq = output_tokenizer.texts_to_sequences(train_data['jpn'])
output_input_integer_seq = output_tokenizer.texts_to_sequences(train_data['eng'])

word2idx_outputs = output_tokenizer.word_index
ic('Total unique words in the output: %s' % len(word2idx_outputs))

num_words_output = len(word2idx_outputs) + 1
max_out_len = max(len(sen) for sen in output_integer_seq)
ic("Length of longest sentence in the output: %g" % max_out_len)

decoder_input_sequences = pad_sequences(output_input_integer_seq, maxlen=max_out_len, padding='post')
ic("decoder_input_sequences.shape:", decoder_input_sequences.shape)

decoder_output_sequences = pad_sequences(output_integer_seq, maxlen=max_out_len, padding='post')
ic("decoder_output_sequences.shape:", decoder_output_sequences.shape)

num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_matrix = zeros((num_words, EMBEDDING_SIZE))
for word, index in word2idx_inputs.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

embedding_layer = Embedding(num_words, EMBEDDING_SIZE, weights=[embedding_matrix])

encoder_inputs = Input(shape=(max_input_len,))
x = embedding_layer(encoder_inputs)
encoder = LSTM(LSTM_NODES, return_state=True)

encoder_outputs, h, c = encoder(x)
encoder_states = [h, c]

decoder_inputs = Input(shape=(max_out_len,))
decoder_embedding = Embedding(num_words_output, LSTM_NODES)
decoder_inputs_x = decoder_embedding(decoder_inputs)

decoder_lstm = LSTM(LSTM_NODES, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)

decoder_dense = Dense(num_words_output-1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(
    optimizer='SGD',
    loss='CategoricalCrossentropy',
    metrics=['accuracy']
)
model.summary()

es = EarlyStopping(monitor='loss', mode='auto')

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

encoder_input_sequences_gpu = cp.asarray(encoder_input_sequences)
decoder_input_sequences_gpu = cp.asarray(decoder_input_sequences)
def createList(r1, r2):
    """
    Create a list of numbers from r1 to r2.

    Parameters
    ----------
    r1 : int
        The start of the range.
    r2 : int
        The end of the range.

    Returns
    -------
    list
        A list of numbers from r1 to r2.

    Examples
    --------
    >>> createList(1, 5)
    [1, 2, 3, 4, 5]
    """
    return list(range(r1, r2+1))
def loopthrough_model(loopamount):
    """
    Train the model in chunks of SIZE_OF_BLOCK to avoid memory errors.

    Args:
        loopamount (int): The number of chunks to split the data into.

    Returns:
        history (list): A list of the history of the model for each chunk.
    """
    history=[]
    for seg in range(loopamount):
        ic(f"training segment {seg} of {loopamount}")
        list_block=createList(seg*SIZE_OF_BLOCK, (seg+1)*SIZE_OF_BLOCK-1)
        train_seg=train_data.select(list_block)
        ic(f"train_seg length: {len(train_seg['eng'])}")
        if len(train_seg['eng'])==0:
            ic("No data in the segment, skipping")
            continue
        encoder_input_sequences_gpu_seg=encoder_input_sequences_gpu[seg*SIZE_OF_BLOCK:(seg+1)*SIZE_OF_BLOCK]
        ic(f"encoder_input_sequences_gpu_seg shape: {encoder_input_sequences_gpu_seg.shape}")
        decoder_input_sequences_gpu_seg=decoder_input_sequences_gpu[seg*SIZE_OF_BLOCK:(seg+1)*SIZE_OF_BLOCK]
        ic(f"decoder_input_sequences_gpu_seg shape: {decoder_input_sequences_gpu_seg.shape}")
        decoder_targets_one_hot = cp.zeros((len(train_seg['eng']), max_out_len,  len(output_tokenizer.word_index)), dtype='float')
        ic(f"decoder_targets_one_hot shape: {decoder_targets_one_hot.shape}")
        decoder_onehot=onehot_loop(SIZE_OF_BLOCK,decoder_output_sequences,decoder_targets_one_hot)
        ic(f"decoder_onehot shape: {decoder_onehot.shape}")
        if decoder_onehot is None:
            ic("decoder_onehot is None, skipping")
            continue
        #ic(len(encoder_input_sequences_gpu),'encoder_input_sequences shape')
        ic(len(decoder_input_sequences_gpu),'decoder_input_sequences shape')
        ic(len(decoder_onehot),'decoder_onehot shape')
        try:
            history.append(model.fit([encoder_input_sequences_gpu_seg.get(), decoder_input_sequences_gpu_seg.get()], decoder_onehot.get(),
                batch_size=BATCH_SIZE,
                epochs=25,
                callbacks=[es]
            ))
        except Exception as e:
            ic(f"Exception occurred: {e}")
        model.save('smaller.keras')
    return history
def onehot_loop(SIZE_OF_BLOCK, decoder_output_sequences, decoder_targets_one_hot):
    """
    This function creates one-hot vectors for the decoder output sequences.
W
    Args:
        SIZE_OF_BLOCK (int): The size of the block to be processed.
        decoder_output_sequences (list of lists): The list of output sequences.
        decoder_targets_one_hot (cupy array): The array to be filled with one-hot vectors.

    Returns:
        cupy array: The array with one-hot vectors.
    """
    size = 0
    for i, d in enumerate(decoder_output_sequences):  # Iterate over the output sequences
        if size != SIZE_OF_BLOCK:  # Check if we have reached the end of the block
            size += 1
            for t, word in enumerate(d):  # Iterate over the words in the sequence
                decoder_targets_one_hot[i, t, word] = 1  # Set the one-hot vector
        else:
            break
    return decoder_targets_one_hot

def lower_block_size_as_needed(SIZE_OF_BLOCK):
    """
    Train the model in chunks of SIZE_OF_BLOCK to avoid memory errors.

    Args:
        SIZE_OF_BLOCK (int): The size of the block to be processed.

    Returns:
        history (list): A list of the history of the model for each chunk.
    """
    # Continue training the model in chunks until the block size is 0
    while SIZE_OF_BLOCK > 0:
        try:
            # Try to train the model with the current block size
            history = loopthrough_model(int(1936097 / SIZE_OF_BLOCK))
            return history
        except MemoryError:
            # If there is a memory error, lower the block size and try again
            SIZE_OF_BLOCK -= 1
            if SIZE_OF_BLOCK <= 0:
                raise RuntimeError("Unable to allocate memory for any block size") from None
            # Print a message to let the user know what is happening
            print('-' * 50)
            print(f"There was a memory error restarting model with block size of {SIZE_OF_BLOCK}")
            print('-' * 50)
    raise ValueError("Block size must be positive")
lower_block_size_as_needed(SIZE_OF_BLOCK)
model.save('translation.keras')
