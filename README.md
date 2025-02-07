# Neural Machine Translation (NMT) Model for Japanese to English Translation

This repository contains a Neural Machine Translation (NMT) model implemented using TensorFlow and Keras. The model is designed to translate sentences from Japanese to English. It utilizes a sequence-to-sequence (Seq2Seq) architecture with Long Short-Term Memory (LSTM) networks and pre-trained GloVe word embeddings.

**Research Question:**

*How can a sequence-to-sequence (Seq2Seq) model with LSTM networks and pre-trained GloVe embeddings be optimized for accurate and efficient Japanese-to-English translation, and what are the key factors influencing its performance on large-scale datasets?*

This research question aims to explore:
1. The effectiveness of LSTM-based Seq2Seq architectures for Japanese-to-English translation.
2. The role of pre-trained word embeddings (e.g., GloVe) in improving translation quality.
3. The challenges of training on large-scale datasets, including memory management and computational efficiency.
4. The impact of hyperparameters (e.g., batch size, embedding size, LSTM nodes) and training strategies (e.g., chunked training, early stopping) on model performance.

By addressing this question, the project seeks to contribute to the development of robust neural machine translation systems for low-resource language pairs like Japanese and English.

## Overview

The code in this repository performs the following tasks:

1. **Environment Setup**: Configures the environment to optimize TensorFlow performance and suppress unnecessary logs.
2. **Data Loading**: Loads a pre-processed dataset containing Japanese-English sentence pairs.
3. **Text Preprocessing**: Tokenizes the input and output sentences, converts them into sequences of integers, and pads them to ensure uniform length.
4. **Embedding Layer**: Initializes an embedding layer using pre-trained GloVe word vectors.
5. **Model Architecture**: Constructs an encoder-decoder model with LSTM layers for sequence processing.
6. **Training**: Trains the model in chunks to avoid memory errors, using a custom training loop.
7. **Model Saving**: Saves the trained model for future use.

## Requirements

To run this code, you need the following libraries and tools:

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- CuPy (for GPU acceleration)
- PyTorch (for GPU memory monitoring)
- Hugging Face `datasets` library
- Matplotlib (for visualization, though not used in the current script)
- Icecream (for debugging and logging)

## File Structure

- **`glove/`**: Directory containing the GloVe word embeddings file (`glove.6B.200d.txt`).
- **`dataset_jpn_eng.hf/`**: Directory containing the pre-processed dataset split into training, validation, and test sets.
- **`translation.keras`**: The saved model after training.
- **`smaller.keras`**: Intermediate model checkpoints saved during training.

## Key Components

### 1. Environment Setup

The script starts by configuring the environment to optimize TensorFlow performance and suppress unnecessary logs. It also checks for GPU availability and prints GPU memory usage if a GPU is available.

```python
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

### 2. Data Loading

The dataset is loaded using the Hugging Face `datasets` library. The dataset is expected to be pre-processed and split into training, validation, and test sets.

```python
train_data = datasets.load_from_disk("dataset_jpn_eng.hf/train")
test_data = datasets.load_from_disk("dataset_jpn_eng.hf/test")
valid_data = datasets.load_from_disk("dataset_jpn_eng.hf/valid")
```

### 3. Text Preprocessing

The input (English) and output (Japanese) sentences are tokenized and converted into sequences of integers. The sequences are then padded to ensure uniform length.

```python
input_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)
input_tokenizer.fit_on_texts(train_data['eng'])
input_integer_seq = input_tokenizer.texts_to_sequences(train_data['eng'])
```

### 4. Embedding Layer

An embedding layer is initialized using pre-trained GloVe word vectors. This layer maps each word in the input vocabulary to a dense vector of fixed size.

```python
embedding_matrix = zeros((num_words, EMBEDDING_SIZE))
for word, index in word2idx_inputs.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
```

### 5. Model Architecture

The model consists of an encoder and a decoder, both implemented using LSTM layers. The encoder processes the input sequence and produces a context vector, which is then used by the decoder to generate the output sequence.

```python
encoder_inputs = Input(shape=(max_input_len,))
x = embedding_layer(encoder_inputs)
encoder = LSTM(LSTM_NODES, return_state=True)
encoder_outputs, h, c = encoder(x)
encoder_states = [h, c]
```

### 6. Training

The model is trained in chunks to avoid memory errors. The training loop processes the data in blocks and saves intermediate model checkpoints.

```python
def loopthrough_model(loopamount):
    history = []
    for seg in range(loopamount):
        ic(f"training segment {seg} of {loopamount}")
        list_block = createList(seg * SIZE_OF_BLOCK, (seg + 1) * SIZE_OF_BLOCK - 1)
        train_seg = train_data.select(list_block)
        ...
```

### 7. Model Saving

After training, the final model is saved to a file for future use.

```python
model.save('translation.keras')
```

## Usage

To train the model, simply run the script. Ensure that the required datasets and GloVe embeddings are in the correct directories.

```bash
python train_translation_model.py
```

## Notes

- The script is designed to handle large datasets by processing them in smaller chunks to avoid memory errors.
- The model uses pre-trained GloVe embeddings, which should be downloaded and placed in the `glove/` directory.
- The dataset should be pre-processed and split into training, validation, and test sets before running the script.

## Future Work

- Implement attention mechanisms to improve translation quality.
- Experiment with different architectures, such as Transformers.
- Add evaluation metrics (e.g., BLEU score) to assess model performance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The GloVe embeddings used in this project were developed by Stanford University.
- The dataset used in this project is from the Hugging Face `datasets` library.

---

This `README.md` provides a comprehensive overview of the code, its components, and how to use it. It also outlines potential future improvements and acknowledges the resources used in the project.
