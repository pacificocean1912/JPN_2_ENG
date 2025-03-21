import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pickle
import datasets
# Define the model architecture (must match the training architecture)
class Seq2SeqModel(nn.Module):
    def __init__(self, embedding_size, lstm_nodes, num_words_output, max_out_len):
        super(Seq2SeqModel, self).__init__()
        self.encoder_lstm = nn.LSTM(embedding_size, lstm_nodes, batch_first=True)
        self.decoder_embedding = nn.Embedding(num_words_output, lstm_nodes)
        self.decoder_lstm = nn.LSTM(lstm_nodes, lstm_nodes, batch_first=True)
        self.decoder_dense = nn.Linear(lstm_nodes, num_words_output)

    def forward(self, encoder_inputs, decoder_inputs):
        encoder_outputs, (h, c) = self.encoder_lstm(encoder_inputs)
        decoder_inputs_x = self.decoder_embedding(decoder_inputs)
        decoder_outputs, _ = self.decoder_lstm(decoder_inputs_x, (h, c))
        decoder_outputs = self.decoder_dense(decoder_outputs)
        return decoder_outputs
def load_dataset() -> tuple:
    """
    Load the dataset from disk.

    Returns:
        tuple: A tuple of train, test and validation datasets.
    """
    try:
        # Load the dataset from disk
        train_data = 0
        test_data = datasets.load_from_disk("dataset_jpn_eng.hf/test")
        valid_data = datasets.load_from_disk("dataset_jpn_eng.hf/valid")
        #train_data = train_data.shuffle(seed=30).select(range(NUM_SENTENCES))
        # Return the datasets
        return train_data, test_data, valid_data
    except Exception as e:
        raise RuntimeError("Error loading datasets: {}".format(e))
train_data, test_data, valid_data = load_dataset()

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters (must match the training configuration)
EMBEDDING_SIZE = 768
LSTM_NODES = 256
MAX_SENTENCE_LENGTH = 50
MAX_OUT_LENGTH = 28
class Tokenizer:
    def __init__(self):
        self.word2idx = {}  # Maps words to indices
        self.idx2word = {}  # Maps indices to words
        self.idx = 1  # Start indexing from 1 (0 is reserved for padding)

    def fit_on_texts(self, texts):
        """
        Fits the tokenizer on a list of texts.
        """
        for text in texts:
            for word in text.split():
                if word not in self.word2idx:
                    self.word2idx[word] = self.idx
                    self.idx2word[self.idx] = word
                    self.idx += 1

    def texts_to_sequences(self, texts):
        """
        Converts a list of texts to sequences of word indices.
        """
        sequences = []
        for text in texts:
            sequence = []
            for word in text.split():
                sequence.append(self.word2idx.get(word, 0))  # 0 for unknown words
            sequences.append(sequence)
        return sequences
# Load the output tokenizer
with open('output_tokenizer.pkl', 'rb') as f:
    output_tokenizer = pickle.load(f)

# Calculate the vocabulary size
num_words_output = len(output_tokenizer.word2idx) + 1  # +1 for padding token

# Initialize the model
model = Seq2SeqModel(EMBEDDING_SIZE, LSTM_NODES, num_words_output, MAX_OUT_LENGTH).to(device)

# Load the saved model weights
model.load_state_dict(torch.load('translation.pth'))
model.eval()  # Set the model to evaluation mode

# Load the BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

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
        if word_index == 0:  # Skip padding tokens
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
def clean(data):
    data = data.replace('<eos>', '')
    data = data.replace('<sos>', '')
    data = data.replace('< sos >', '')
    data = data.replace('questionmark', '?')
    return data
print(test_data)
# Test the model with example sentences
test_sentences =0 

for i in range(100):
    en_sentence = test_data[i]['eng']
    en_sentence = clean(en_sentence)
    jp_sentence = test_data[i]['jpn']
    jp_sentence = clean(jp_sentence)
    translated_sentence = translate_sentence(en_sentence)
    translated_sentence = clean(translated_sentence)
    
    print(f"Input: {en_sentence}")
    print(f"Translation: {translated_sentence}")
    print(f"Original: {jp_sentence}")
    print("-" * 50)