import os
# Disable oneDNN optimizations to avoid potential issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress TensorFlow logging (only show errors)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Allow duplicate libraries to avoid runtime errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import necessary libraries
try:
    from icecream import ic  # For debugging and logging 
    #sadly the hpc doens't have icecream installed so I can't use it so I replaced it with print
except Exception as e:
    print("icecream not found",e)
try:    
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    print("can't find pytorch",e)
try:
    import numpy as np
except Exception as e:
    print("numpy needs to be installed ",e)
try:
    import pickle  # For saving/loading data
except Exception as e:
    print("pickle needs to be installed ",e)
try:
    from datasets import load_dataset  # For loading datasets
except Exception as e:
    print("datasets needs to be installed ",e)
try:
    from transformers import BertTokenizer, BertModel  # For BERT embeddings
except Exception as e:
    print("BertTokenizer needs to be installed ",e)
try:
    from torch.nn.utils.rnn import pad_sequence  # For padding sequences
except Exception as e:
    print("torch.nn.utils.rnn needs to be installed ",e)
try:
    from torch.optim.lr_scheduler import StepLR  # For learning rate scheduling
except Exception as e:
    print("torch.optim.lr_scheduler needs to be installed ",e)
try:    
    import datasets
except Exception as e:
    print("datasets needs to be installed ",e)
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")

print("starting the program")
# Set the device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters and constants
BATCH_SIZE = 64  # Number of samples per batch
EPOCHS = 20  # Number of training epochs
LSTM_NODES = 256  # Number of LSTM units
NUM_SENTENCES = 7000  # Maximum number of sentences to use
MAX_SENTENCE_LENGTH = 2000  # Maximum length of a sentence
MAX_NUM_WORDS = 200000  # Maximum number of words in the vocabulary
EMBEDDING_SIZE = 768  # Size of BERT embeddings
SIZE_OF_BLOCK = 10000  # Size of data chunks for training
EPOCHS = 200

# Load BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
# Load the dataset from disk
def load_dataset() -> tuple:
    """
    Load the dataset from disk.

    Returns:
        tuple: A tuple of train, test and validation datasets.
    """
    try:
        # Load the dataset from disk
        train_data = datasets.load_from_disk("dataset_jpn_eng.hf/train")
        test_data = datasets.load_from_disk("dataset_jpn_eng.hf/test")
        valid_data = datasets.load_from_disk("dataset_jpn_eng.hf/valid")
        train_data = train_data.shuffle(seed=43).select(range(NUM_SENTENCES))
        print("train_data.shape:", train_data.shape)
        print(train_data)
        # Return the datasets
        return train_data, test_data, valid_data
    except Exception as e:
        raise RuntimeError("Error loading datasets: {}".format(e))
train_data, test_data, valid_data = load_dataset()

# Tokenize the input English sentences using BERT
def bert_tokenize(sentences):
    return bert_tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

print("starting tokenizing")
input_encodings = bert_tokenize(train_data['eng'])
print("input_encodings")
input_ids = input_encodings['input_ids'].to(device)
#print("input_ids.shape:", input_ids.shape)
attention_mask = input_encodings['attention_mask'].to(device)
#print("attention_mask.shape:", attention_mask.shape)
MAX_SENTENCE_LENGTH = input_ids.shape[1]
#print("input_ids[0]:", input_ids[1])

# Get BERT embeddings for the input sentences
with torch.no_grad():
    bert_embeddings = bert_model(input_ids, attention_mask=attention_mask).last_hidden_state
print("bert_embeddings.shape:", bert_embeddings.shape)

# Tokenize the output Japanese sentences
class Tokenizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_texts(self, texts):
        for text in texts:
            for word in text.split():
                if word not in self.word2idx:
                    self.word2idx[word] = self.idx
                    self.idx2word[self.idx] = word
                    self.idx += 1

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = []
            for word in text.split():
                sequence.append(self.word2idx.get(word, 0))
            sequences.append(sequence)
        return sequences

output_tokenizer = Tokenizer()
output_tokenizer.fit_on_texts(train_data['jpn'] + train_data['eng'])
output_integer_seq = output_tokenizer.texts_to_sequences(train_data['jpn'])
output_input_integer_seq = output_tokenizer.texts_to_sequences(train_data['eng'])

# Get the word-to-index mapping for output
word2idx_outputs = output_tokenizer.word2idx
print('Total unique words in the output: %s' % len(word2idx_outputs))

# Calculate the number of words in the output vocabulary
num_words_output = len(word2idx_outputs) + 1
# Find the length of the longest sentence in the output
max_out_len = max(len(sen) for sen in output_integer_seq)
print("Length of longest sentence in the output: %g" % max_out_len)

# Pad the decoder input sequences to the maximum length
def pad_sequences(sequences, maxlen, padding='post'):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < maxlen:
            if padding == 'post':
                padded_seq = seq + [0] * (maxlen - len(seq))
            else:
                padded_seq = [0] * (maxlen - len(seq)) + seq
        else:
            padded_seq = seq[:maxlen]
        padded_sequences.append(padded_seq)
    return torch.tensor(padded_sequences, dtype=torch.long)

decoder_input_sequences = pad_sequences(output_input_integer_seq, maxlen=max_out_len, padding='post')
print("decoder_input_sequences.shape:", decoder_input_sequences.shape)

# Pad the decoder output sequences to the maximum length
decoder_output_sequences = pad_sequences(output_integer_seq, maxlen=max_out_len, padding='post')
print("decoder_output_sequences.shape:", decoder_output_sequences.shape)

# Define the model
class Seq2SeqModel(nn.Module):
    def __init__(self, embedding_size, lstm_nodes, num_words_output, max_out_len):
        super(Seq2SeqModel, self).__init__()
        self.encoder_lstm = nn.LSTM(embedding_size, lstm_nodes, batch_first=True)
        self.decoder_embedding = nn.Embedding(num_words_output, lstm_nodes)
        self.decoder_lstm = nn.LSTM(lstm_nodes, lstm_nodes, batch_first=True)
        self.decoder_dense = nn.Linear(lstm_nodes, num_words_output)

    def forward(self,encoder_inputs, decoder_inputs):
        _, (h, c) = self.encoder_lstm(encoder_inputs)
        decoder_inputs_x = self.decoder_embedding(decoder_inputs)
        decoder_outputs, _ = self.decoder_lstm(decoder_inputs_x, (h, c))
        decoder_outputs = self.decoder_dense(decoder_outputs)
        return decoder_outputs
try:
    saved_state_dict = torch.load('smaller.pth')
except FileNotFoundError:
    print("No saved model found.")

model = Seq2SeqModel(EMBEDDING_SIZE, LSTM_NODES, num_words_output, max_out_len).to(device)
old_num_words = saved_state_dict['decoder_embedding.weight'].shape[0]
new_num_words = num_words_output


# Resize the embedding layer
embedding_weight = saved_state_dict['decoder_embedding.weight']
new_embedding_weight = torch.zeros((new_num_words, LSTM_NODES)).to(device)
new_embedding_weight[:old_num_words] = embedding_weight  # Copy old weights
saved_state_dict['decoder_embedding.weight'] = new_embedding_weight

# Resize the dense layer
dense_weight = saved_state_dict['decoder_dense.weight']
new_dense_weight = torch.zeros((new_num_words, LSTM_NODES)).to(device)
new_dense_weight[:old_num_words] = dense_weight  # Copy old weights
saved_state_dict['decoder_dense.weight'] = new_dense_weight

dense_bias = saved_state_dict['decoder_dense.bias']
new_dense_bias = torch.zeros(new_num_words).to(device)
new_dense_bias[:old_num_words] = dense_bias  # Copy old weights
saved_state_dict['decoder_dense.bias'] = new_dense_bias

# Load the resized weights into the model
model.load_state_dict(saved_state_dict)
model.eval()

# Compile the model with Adam optimizer and CrossEntropyLoss
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Define early stopping callback
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = None

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = current_loss
            self.counter = 0
        return False

es = EarlyStopping(patience=5)

# Convert input sequences to PyTorch tensors for GPU processing
encoder_input_sequences_gpu = bert_embeddings
decoder_input_sequences_gpu = decoder_input_sequences.to(device)

#-------------------- running the project -----------------------------------------------------------------------------------

#try:
#model.load_state_dict(torch.load('smaller.pth'))
#except Exception as e:
#    print("model failed to load",e)
def loopthrough_model():
    print(f"train length: {len(train_data['eng'])}")

    # Convert decoder_output_sequences to a PyTorch tensor
    decoder_output_sequences_tensor = torch.tensor(decoder_output_sequences, dtype=torch.long).to(device)
    print(f"decoder_output_sequences_tensor shape: {decoder_output_sequences_tensor.shape}")

    # Train the model on the segment
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(encoder_input_sequences_gpu, decoder_input_sequences_gpu)
        #print(f"outputs shape: {outputs.shape}")

        # Reshape outputs and targets for CrossEntropyLoss
        outputs_reshaped = outputs.view(-1, num_words_output)  # Shape: (batch_size * sequence_length, num_words_output)
        targets_reshaped = decoder_output_sequences_tensor.view(-1)  # Shape: (batch_size * sequence_length)

        #print(f"outputs_reshaped shape: {outputs_reshaped.shape}")
        #print(f"targets_reshaped shape: {targets_reshaped.shape}")

        # Compute loss
                # Compute loss
        loss = criterion(outputs_reshaped, targets_reshaped)

        # Calculate accuracy
        _, predicted_indices = torch.max(outputs_reshaped, 1)  # Get predicted token indices
        correct_predictions = (predicted_indices == targets_reshaped).sum().item()
        total_predictions = targets_reshaped.size(0)
        accuracy = correct_predictions / total_predictions

        print(f"Epoch {epoch+1}/{EPOCHS}, Accuracy: {accuracy * 100:.2f}% ,Loss: {loss.item()}" )


        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Early stopping check
        if es(loss.item()):
            print("Early stopping")
            break

    # Save the model
    torch.save(model.state_dict(), 'smaller.pth')

def onehot_loop(decoder_output_sequences, decoder_targets_one_hot):
    """
    This function creates one-hot vectors for the decoder output sequences.

    Args:
        decoder_output_sequences (list of lists): The list of output sequences.
        decoder_targets_one_hot (torch.Tensor): The tensor to be filled with one-hot vectors.

    Returns:
        torch.Tensor: The tensor with one-hot vectors.
    """
    for i, d in enumerate(decoder_output_sequences):
        for t, word in enumerate(d):
            decoder_targets_one_hot[i, t, word] = 1
    return decoder_targets_one_hot

loopthrough_model()
# Save the final model
torch.save(model.state_dict(), 'translation.pth')

def test_model():
    # Load the BERT tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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
            torch.Tensor: The BERT embeddings for the input sentence.
        """
        # Tokenize the input text using BERT
        input_encodings = bert_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=MAX_SENTENCE_LENGTH,
            return_tensors="pt"
        )
        input_ids = input_encodings['input_ids'].to(device)
        attention_mask = input_encodings['attention_mask'].to(device)

        # Get BERT embeddings for the input sentence
        with torch.no_grad():
            bert_embeddings = bert_model(input_ids, attention_mask=attention_mask).last_hidden_state
        return bert_embeddings

    # Function to decode the model's output
    def decode_output(sequence):
        """
        Decode the model's output sequence into a sentence.

        Args:
            sequence (torch.Tensor): The output sequence from the model.

        Returns:
            str: The decoded sentence.
        """
        # Convert the sequence of word indices to words
        words = []
        for word_index in sequence:
            if word_index == 0:
                continue
            word = output_tokenizer.idx2word.get(word_index.item(), '')
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
        decoder_input = torch.zeros((1, MAX_OUT_LENGTH), dtype=torch.long).to(device)
        decoder_input[0, 0] = output_tokenizer.word2idx.get('<start>', 0)

        # Generate the translation word by word
        for i in range(1, MAX_OUT_LENGTH):
            # Predict the next word
            with torch.no_grad():
                predictions = model(encoder_input, decoder_input)
            predicted_word_index = torch.argmax(predictions[0, i-1, :]).item()

            # Stop if the end token is predicted
            if predicted_word_index == output_tokenizer.word2idx.get('<end>', 0):
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
