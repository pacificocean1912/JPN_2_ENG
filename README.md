# Neural Machine Translation (NMT) Model for Japanese to English Translation

# Neural Machine Translation (NMT) Model with BERT and LSTM

This repository contains a Neural Machine Translation (NMT) model implemented using PyTorch, designed to translate sentences between English and Japanese. The model features a sequence-to-sequence (Seq2Seq) architecture with LSTM networks and leverages pre-trained BERT embeddings for enhanced performance.

## Key Features

- **BERT-based encoder**: Utilizes pre-trained BERT model for high-quality sentence embeddings
- **LSTM-based decoder**: Generates translations with sequential processing
- **Dynamic vocabulary handling**: Automatically adjusts for new words in the vocabulary
- **GPU acceleration**: Fully optimized for CUDA-enabled GPUs
- **Memory-efficient training**: Processes data in manageable chunks
- **Model persistence**: Saves trained models and tokenizers for deployment

## Research Focus

**Primary Question**:  
How can a BERT-enhanced Seq2Seq model with LSTM networks be optimized for efficient and accurate bidirectional translation between Japanese and English?

**Key Investigation Areas**:
- Effectiveness of BERT embeddings versus traditional word embeddings
- Impact of dynamic vocabulary resizing on model performance
- Optimization strategies for large-scale translation tasks
- Hyperparameter tuning for LSTM-based translation models

## Technical Specifications

### Architecture Overview
- **Encoder**: BERT model (bert-base-uncased) for input sequence processing
- **Decoder**: Two-layer LSTM network with attention mechanism
- **Embedding**: 768-dimensional BERT embeddings (fixed) + trainable decoder embeddings

### Key Components
1. **BERT Tokenization**: 
   ```python
   bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

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
