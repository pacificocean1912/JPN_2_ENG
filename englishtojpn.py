import os
# Disable oneDNN optimizations to avoid potential issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress TensorFlow logging (only show errors)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Allow duplicate libraries to avoid runtime errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import necessary libraries
from icecream import ic  # For debugging and logging
import tensorflow as tf
from keras.models import Model  # For creating the model
from keras.layers import Input, LSTM, Dense, Embedding  # Layers for the model
from keras.preprocessing.sequence import pad_sequences  # For padding sequences
import numpy as np
import pickle  # For saving/loading data
from keras.callbacks import EarlyStopping  # For early stopping during training
import datasets  # For loading datasets
from datasets import load_dataset  # For loading datasets
import torch  # For GPU support
from numpy import asarray, zeros  # For array operations
import cupy as cp  # For GPU-accelerated array operations
from transformers import BertTokenizer, TFBertModel  # For BERT embeddings
ic("starting the program")
# Set the device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters and constants
BATCH_SIZE = 64  # Number of samples per batch
EPOCHS = 20  # Number of training epochs
LSTM_NODES = 256  # Number of LSTM units
NUM_SENTENCES = 2000  # Maximum number of sentences to use
MAX_SENTENCE_LENGTH = 118  # Maximum length of a sentence
MAX_NUM_WORDS = 200000  # Maximum number of words in the vocabulary
EMBEDDING_SIZE = 768  # Size of BERT embeddings
SIZE_OF_BLOCK = 10000  # Size of data chunks for training
embeddings_dictionary = dict()  # Dictionary to store GloVe embeddings
# Configure TensorFlow to allow GPU memory growth
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# Load BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Load the dataset from disk
def load_dataset() -> tuple:
    """
    Load the dataset from disk.

    Returns:
        tuple: A tuple of train, test and validation datasets.
    """
    try:
        # Load the dataset from disk
        ic("loading dataset")
        train_data = datasets.load_from_disk("dataset_jpn_eng.hf/train")
        test_data = datasets.load_from_disk("dataset_jpn_eng.hf/test")
        valid_data = datasets.load_from_disk("dataset_jpn_eng.hf/valid")
        train_data = train_data[:NUM_SENTENCES]
        ic(len(train_data))
        # Return the datasets
        return train_data, test_data, valid_data
    except Exception as e:
        raise RuntimeError("Error loading datasets: {}".format(e))
train_data, test_data, valid_data = load_dataset()

# Tokenize the input English sentences using BERT
def bert_tokenize(sentences):
    return bert_tokenizer(sentences, padding=True, truncation=True, return_tensors="tf")
ic("tokenizing")
input_encodings = bert_tokenize(train_data['eng'])
input_ids = input_encodings['input_ids']
attention_mask = input_encodings['attention_mask']
ic("input_ids.shape:", input_ids.shape)
ic("attention_mask.shape:", attention_mask.shape)
ic("input_ids[0]:", input_ids[0])
# Get BERT embeddings for the input sentences
bert_embeddings = bert_model(input_ids, attention_mask=attention_mask).last_hidden_state
ic("bert_embeddings.shape:", bert_embeddings.shape)
# Tokenize the output Japanese sentences
output_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS, filters='')
output_tokenizer.fit_on_texts(train_data['jpn'] + train_data['eng'])  # Fit tokenizer on Japanese and English sentences
output_integer_seq = output_tokenizer.texts_to_sequences(train_data['jpn'])  # Convert Japanese sentences to sequences
output_input_integer_seq = output_tokenizer.texts_to_sequences(train_data['eng'])  # Convert English sentences to sequences

# Get the word-to-index mapping for output
word2idx_outputs = output_tokenizer.word_index
ic('Total unique words in the output: %s' % len(word2idx_outputs))  # Log the number of unique words

# Calculate the number of words in the output vocabulary
num_words_output = len(word2idx_outputs) + 1
# Find the length of the longest sentence in the output
max_out_len = max(len(sen) for sen in output_integer_seq)
ic("Length of longest sentence in the output: %g" % max_out_len)  # Log the longest sentence length

# Pad the decoder input sequences to the maximum length
decoder_input_sequences = pad_sequences(output_input_integer_seq, maxlen=max_out_len, padding='post')
ic("decoder_input_sequences.shape:", decoder_input_sequences.shape)  # Log the shape of padded sequences

# Pad the decoder output sequences to the maximum length
decoder_output_sequences = pad_sequences(output_integer_seq, maxlen=max_out_len, padding='post')
ic("decoder_output_sequences.shape:", decoder_output_sequences.shape)  # Log the shape of padded sequences

# Define the encoder inputs
encoder_inputs = Input(shape=(MAX_SENTENCE_LENGTH, EMBEDDING_SIZE))
encoder = LSTM(LSTM_NODES, return_state=True)  # Define the LSTM encoder

# Get the encoder outputs and states
encoder_outputs, h, c = encoder(encoder_inputs)
encoder_states = [h, c]  # Store the hidden and cell states

# Define the decoder inputs
decoder_inputs = Input(shape=(max_out_len,))
decoder_embedding = Embedding(num_words_output, LSTM_NODES)  # Define the embedding layer for the decoder
decoder_inputs_x = decoder_embedding(decoder_inputs)  # Pass inputs through the embedding layer

# Define the LSTM decoder
decoder_lstm = LSTM(LSTM_NODES, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)  # Pass inputs through the LSTM

# Define the dense layer for the decoder outputs
decoder_dense = Dense(num_words_output-1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)  # Pass outputs through the dense layer
#model=tf.keras.models.load_model('translation.keras')

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model with SGD optimizer and categorical crossentropy loss
model.compile(
    optimizer='rmsprop',
    loss='CategoricalCrossentropy',
    metrics=['accuracy']
)
model.summary()  # Print the model summary
model.load_weights('translation.keras')
# Define early stopping callback
es = EarlyStopping(monitor='loss', mode='auto')
# Convert input sequences to CuPy arrays for GPU processing
encoder_input_sequences_gpu = cp.asarray(bert_embeddings.numpy())
decoder_input_sequences_gpu = cp.asarray(decoder_input_sequences)

#-------------------- running the project -----------------------------------------------------------------------------------

def loopthrough_model():
    train_seg=train_data
    ic(f"train length: {len(train_seg['eng'])}")  # Log the length of the segment
    decoder_targets_one_hot = cp.zeros((len(train_seg['eng']), max_out_len,  len(output_tokenizer.word_index)), dtype='float')  # Initialize one-hot vectors
    ic(f"decoder_targets_one_hot shape: {decoder_targets_one_hot.shape}")  # Log the shape of the one-hot vectors

    decoder_onehot=onehot_loop(decoder_output_sequences,decoder_targets_one_hot)  # Create one-hot vectors for the segment
         
    ic(f"decoder_onehot shape: {decoder_onehot.shape}")  # Log the shape of the one-hot vectors
    if decoder_onehot is None:
        ic("decoder_onehot is None, skipping")  # Skip if one-hot vectors are not created

    ic(len(decoder_input_sequences_gpu),'decoder_input_sequences shape')  # Log the shape of the decoder inputs
    ic(len(decoder_onehot),'decoder_onehot shape')  # Log the shape of the one-hot vectors
     # Train the model on the segment
    model.fit([encoder_input_sequences_gpu.get(), decoder_input_sequences_gpu.get()], decoder_onehot.get(),
                batch_size=BATCH_SIZE,
                epochs=100,
                callbacks=[es]
            )
    
    model.save('smaller.keras')  # Save the model after each segment

def onehot_loop(decoder_output_sequences, decoder_targets_one_hot):
    """
    This function creates one-hot vectors for the decoder output sequences.

    Args:
        decoder_output_sequences (list of lists): The list of output sequences.
        decoder_targets_one_hot (cupy array): The array to be filled with one-hot vectors.

    Returns:
        cupy array: The array with one-hot vectors.
    """
    for i, d in enumerate(decoder_output_sequences):  # Iterate over the output sequences
        for t, word in enumerate(d):  # Iterate over the words in the sequence
            decoder_targets_one_hot[i, t, word] = 1  # Set the one-hot vector
    return decoder_targets_one_hot
loopthrough_model()
# Save the final model
model.save('translation.keras')
def test_model():    

    # Load the BERT tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load the output tokenizer (used for decoding the output sequences)
    # Assuming you saved the output tokenizer during training

    # Define the maximum sentence length for input and output
    MAX_SENTENCE_LENGTH = 50
    MAX_OUT_LENGTH = 28  # Adjust based on your training data

    # Function to preprocess input text
    def preprocess_input(text):
        """
        Preprocess the input text using BERT tokenization and embedding.

        Args:
            text (str): The input sentence to translate.

        Returns:
            np.array: The BERT embeddings for the input sentence.
        """
        # Tokenize the input text using BERT
        input_encodings = bert_tokenizer(
            text,
            padding='max_length',  # Pad to max length
            truncation=True,       # Truncate to max length
            max_length=MAX_SENTENCE_LENGTH,  # Set max length
            return_tensors="tf"
        )
        input_ids = input_encodings['input_ids']
        attention_mask = input_encodings['attention_mask']

        # Get BERT embeddings for the input sentence
        bert_embeddings = bert_model(input_ids, attention_mask=attention_mask).last_hidden_state
        return bert_embeddings.numpy()

    # Function to decode the model's output
    def decode_output(sequence):
        """
        Decode the model's output sequence into a sentence.

        Args:
            sequence (np.array): The output sequence from the model.

        Returns:
            str: The decoded sentence.
        """
        # Convert the sequence of word indices to words
        words = []
        for word_index in sequence:
            if word_index == 0:  # Skip padding tokens
                continue
            word = output_tokenizer.index_word.get(word_index, '')
            if word:
                words.append(word)
        return ' '.join(words)

    # Function to generate a translation
    def translate_sentence(input_sentence):
        """
        Translate an input sentence using the trained model.

        Args:
            input_sentence (str): The input sentence to translate.

        Returns:
            str: The translated sentence.
        """
        # Preprocess the input sentence
        encoder_input = preprocess_input(input_sentence)

        # Initialize the decoder input with the start token
        decoder_input = np.zeros((1, MAX_OUT_LENGTH), dtype='int32')
        decoder_input[0, 0] = output_tokenizer  # Assuming '<start>' is the start token

        # Generate the translation word by word
        for i in range(1, MAX_OUT_LENGTH):
            # Predict the next word
            predictions = model.predict([encoder_input, decoder_input], verbose=0)
            predicted_word_index = np.argmax(predictions[0, i-1, :])

            # Stop if the end token is predicted
            if predicted_word_index == output_tokenizer.word_index['<end>']:  # Assuming '<end>' is the end token
                break

            # Add the predicted word to the decoder input
            decoder_input[0, i] = predicted_word_index

        # Decode the output sequence
        translated_sentence = decode_output(decoder_input[0])
        return translated_sentence

    # Test the model with an example sentence
    input_sentence = "Hello, how are you?"
    translated_sentence = translate_sentence(input_sentence)
    print(f"Input: {input_sentence}")
    print(f"Translation: {translated_sentence}")
pickle.dump(output_tokenizer, open('output_tokenizer.pkl', 'wb'))
