import time
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from multiprocessing import Pool, cpu_count

# Initialize SBERT model globally
model_sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

def sbert_embed_single_data(text):
    """Embed a single piece of text using SBERT."""
    return model_sbert.encode(text, show_progress_bar=False)

def embed_texts_in_batches(texts, batch_size):
    """
    Embed a list of texts in batches to optimize memory usage and processing time.

    Parameters:
        - texts: List or array of text data.
        - batch_size: Number of texts per batch.

    Returns:
        - Numpy array of embeddings.
    """
    num_cpus = max(1, cpu_count() - 1)  # Reserve one CPU for the main process
    embeddings = []

    # Process texts in parallel batches
    with Pool(num_cpus) as pool:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings.extend(pool.map(sbert_embed_single_data, batch))

    return np.array(embeddings)

def sbert_embed_data(docs_df, queries_df, save_path, lang='ru', batch_size=1000):
    """
    Embed document and query data using SBERT, and save embeddings.

    Parameters:
        - docs_df: DataFrame containing document data.
        - queries_df: DataFrame containing query data.
        - save_path: Directory to save embeddings.
        - lang: Language code for naming files.
        - batch_size: Number of texts to process in each batch.
    """
    os.makedirs(save_path, exist_ok=True)

    start_time = time.time()

    # Embed documents
    print(f"Embedding {len(docs_df)} documents...")
    doc_texts = docs_df["text"].fillna("").values
    doc_embeddings = embed_texts_in_batches(doc_texts, batch_size)

    # Save document embeddings
    doc_embeddings_file = os.path.join(save_path, f"{lang}_doc_embeddings.npy")
    np.save(doc_embeddings_file, doc_embeddings)
    print(f"Document embeddings saved to {doc_embeddings_file}")

    # Embed queries
    if queries_df is not None:
        print(f"Embedding {len(queries_df)} queries...")
        query_texts = queries_df["description"].fillna("").values
        query_embeddings = embed_texts_in_batches(query_texts, batch_size)

        # Save query embeddings
        query_embeddings_file = os.path.join(save_path, f"{lang}_query_embeddings.npy")
        np.save(query_embeddings_file, query_embeddings)
        print(f"Query embeddings saved to {query_embeddings_file}")
    else:
        print("No queries found to embed.")

    elapsed_time = time.time() - start_time
    print(f"Embedding process completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    # File paths and configurations
    filtered_docs_path = './sbert_sentence_transformer/output/ru_filtered_docs.csv'
    queries_path = './sbert_sentence_transformer/output/ru_queries.csv'
    save_path = './sbert_sentence_transformer/output'
    lang = 'ru'
    batch_size = 1000  # Batch size for embedding

    # Load documents and queries
    docs_df = pd.read_csv(filtered_docs_path)
    queries_df = pd.read_csv(queries_path) if os.path.exists(queries_path) else None

    # Embed documents and queries, and save results
    sbert_embed_data(docs_df, queries_df, save_path, lang=lang, batch_size=batch_size)