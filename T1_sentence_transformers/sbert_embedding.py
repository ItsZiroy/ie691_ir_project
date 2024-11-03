import time
from sentence_transformers import SentenceTransformer
from multiprocessing import Pool

# initialize SBERT model
model_sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

def sbert_embed_single_data(text):
    """Embed a single piece of text."""
    return model_sbert.encode(text, show_progress_bar=False)

def sbert_embed_data(data):
    embeddings = {}

    # specify dataset
    lang = 'ru'
    if lang in data:
        docs_df = data[lang]['docs']
        queries_df = data[lang]['queries']

        print(f"\nProcessing embeddings for language: {lang.upper()}")

        start_time = time.time()

        # multi-processing for efficiency
        with Pool() as pool:
            doc_embeddings = pool.map(sbert_embed_single_data, docs_df["text"].fillna("").values)
            query_embeddings = pool.map(sbert_embed_single_data, queries_df["description"].fillna("").values)

        # store embeddings
        embeddings[lang] = {
            'doc_embeddings': doc_embeddings,
            'query_embeddings': query_embeddings
        }

        elapsed_time = time.time() - start_time
        print(f"Embeddings for language {lang.upper()} processed in {elapsed_time:.2f} seconds.\n")

    else:
        print("Russian language data not found in the provided dataset.")

    return embeddings

if __name__ == "__main__":
    from data_preparation import load_datasets, load_data
    datasets = load_datasets()
    data = load_data(datasets, lang='ru', sample_ratio=1.0, max_size=None)  # specify based on comment in data_exploration.py
    embeddings = sbert_embed_data(data)