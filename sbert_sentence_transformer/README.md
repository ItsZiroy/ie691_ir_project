
# SBERT Multilingual Information Retrieval Project

This repository folder provides a complete pipeline for multilingual information retrieval using SBERT (Sentence-BERT). The workflow includes data preparation, embedding generation, document retrieval, and evaluation of retrieval performance.

## Table of Contents

1. [Overview](#overview)
2. [Data Preparation](#data-preparation)
3. [SBERT Embeddings](#sbert-embeddings)
4. [Retrieval](#retrieval)
5. [Evaluation](#evaluation)
6. [Dependencies](#dependencies)
7. [Usage](#usage)

---

## Overview

The project processes multilingual datasets for semantic search. It uses SBERT to generate embeddings and evaluates the quality of document retrieval against predefined relevance scores using IR metrics.

---

## Data Preparation

**Script**: `data_preparation.py`

The script processes datasets using the `ir_datasets` library, filters relevant documents, and saves the results in structured CSV files for further use.

### Key Functions

- **`load_datasets()`**: Loads datasets for various languages (e.g., Russian, Chinese) using `ir_datasets`.
- **`load_data(datasets, lang=None, max_size=None, save_path=None)`**:
  - Filters relevant documents and queries.
  - Saves filtered data (`docs`, `queries`, `qrels`) to CSV.

### Output

Filtered files saved in the specified `save_path`:
- `{lang}_filtered_docs.csv`
- `{lang}_queries.csv`
- `{lang}_filtered_qrels.csv`

## SBERT Embeddings

**Script**: `sbert_embeddings.py`

This script generates embeddings for documents and queries using SBERT, saving the embeddings as `.npy` files for retrieval.

### Key Functions

- **`sbert_embed_single_data(text)`**: Encodes a single text using SBERT.
- **`embed_texts_in_batches(texts, batch_size)`**: Processes texts in parallel batches.
- **`sbert_embed_data(docs_df, queries_df, save_path, lang='ru', batch_size)`**:
  - Embeds document and query data.
  - Saves embeddings as `.npy` files.

### Output

Embeddings saved in the specified `save_path`:
- `{lang}_doc_embeddings.npy`
- `{lang}_query_embeddings.npy`

## Retrieval

**Script**: `retrieval.py`

This script retrieves relevant documents for a given query based on cosine similarity between precomputed SBERT embeddings.

### Key Functions

- **`load_embeddings(save_path, lang)`**: Loads precomputed embeddings.
- **`retrieve_documents(query_embedding, doc_embeddings)`**: Computes cosine similarity for retrieval.
- **`perform_custom_query_retrieval(custom_query, embeddings, lang, data, save_path)`**:
  - Encodes a custom query.
  - Retrieves the most similar documents based on precomputed embeddings.

### Output

Custom query results saved as:
- `{lang}_custom_query_results.csv`

## Evaluation

**Script**: `evaluation.py`

Evaluates retrieval performance using IR metrics such as Precision, Recall, nDCG, etc.

### Key Functions

- Encodes queries and computes document similarities in batches.
- Computes evaluation metrics using `ir_measures`.

## Dependencies

- Python 3.8+
- `ir_datasets`
- `ir_measures`
- `sentence_transformers`
- `numpy`
- `pandas`
- `scikit-learn`

## Usage

1. Prepare data: `data_preparation.py`
2. Generate embeddings: `sbert_embeddings.py`
3. Perform retrieval: `retrieval.py`
4. Evaluate results: `evaluation.py`
