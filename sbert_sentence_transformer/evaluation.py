import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ir_datasets
import ir_measures
from ir_measures import *

# Load the dataset
dataset = ir_datasets.load("neuclir/1/multi/trec-2023")
docs_iter = dataset.docs_iter()  # Use iterator to avoid loading all docs
queries = list(dataset.queries_iter())  # Load all queries into memory
qrels = list(dataset.qrels_iter())  # Load qrels into memory

# Process a limited number of documents (e.g., 50,000) in chunks
doc_batch_size = 50000  # Adjust batch size based on available memory
max_docs = 200000  # Total number of documents to process
processed_docs = 0

# Load the SBERT model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# Encode queries
query_titles = [query.title for query in queries]
query_ids = [query.query_id for query in queries]
query_embeddings = model.encode(query_titles, convert_to_tensor=False)

results = []  # Store results

print(f"Processing up to {max_docs} documents in batches of {doc_batch_size}...")

while processed_docs < max_docs:
    # Load the next batch of documents
    batch_docs = []
    try:
        for _ in range(doc_batch_size):
            batch_docs.append(next(docs_iter))
    except StopIteration:
        break  # Stop when there are no more documents

    print(f"Loaded {len(batch_docs)} documents in this batch.")

    # Convert the batch of documents to a DataFrame
    doc_df = pd.DataFrame([{"doc_id": doc.doc_id, "text": doc.text} for doc in batch_docs])

    # Encode document texts
    doc_embeddings = model.encode(doc_df["text"].tolist(), convert_to_tensor=False)

    # Compute cosine similarity and store top 1000 results
    cos_sim = cosine_similarity(query_embeddings, doc_embeddings)

    for i, sim_scores in enumerate(cos_sim):
        top_n_indices = np.argsort(sim_scores)[::-1][:1000]  # Top 1000 documents
        top_doc_ids = [doc_df.iloc[idx]["doc_id"] for idx in top_n_indices]
        top_scores = sim_scores[top_n_indices]
        for doc_id, score in zip(top_doc_ids, top_scores):
            results.append({"query_id": query_ids[i], "doc_id": doc_id, "score": score})

    processed_docs += len(batch_docs)
    print(f"Processed {processed_docs} documents so far.")

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
print(f"Results DataFrame shape: {results_df.shape}")
print(results_df.head())

# Evaluate using IR measures
qrels_df = pd.DataFrame(qrels)  # Convert qrels to a DataFrame
evaluation_metrics = ir_measures.calc_aggregate(
    [R@1000, R@100, RR@10, AP, nDCG@20, P@5, RBP(rel=1), Judged@10],
    qrels_df[["query_id", "doc_id", "relevance"]],
    results_df
)

# Display evaluation metrics
print("Evaluation Metrics:")
print(evaluation_metrics)
