import pandas as pd
import numpy as np
from data_preparation import load_datasets, load_data
from sbert_embedding import sbert_embed_data
from retrieval import retrieve_documents

def ndcg_at_k(y_true, y_score, k):
    sorted_indices = np.argsort(y_score)[::-1][:k]
    sorted_relevances = y_true[sorted_indices]
    dcg = np.sum(sorted_relevances / np.log2(np.arange(2, len(sorted_relevances) + 2)))
    ideal_dcg = np.sum(np.sort(y_true)[::-1][:k] / np.log2(np.arange(2, k + 2)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0

def evaluate_retrieval(retrieval_results, qrels):
    evaluation_results = []
    for lang, results in retrieval_results.items():
        if not isinstance(lang, str):
            raise ValueError("Expected language code (string) as key in retrieval_results, but found non-string key.")
        
        print(f"\nEvaluating retrieval for language: {lang.upper()}")
        lang_qrels = qrels[lang]  # access qrels for the specific language
        
        for query_id, similarities in results.items():
            # extract relevance labels for the query
            true_relevance = lang_qrels[lang_qrels['query_id'] == query_id]['relevance'].values
            
            # skip if no relevance labels are found for a query id
            if len(true_relevance) == 0:
                print(f"No relevance labels found for query_id {query_id} in language {lang.upper()}. Skipping this query.")
                continue
            
            for k in [5, 10]:
                if len(similarities) >= k:
                    top_k_indices = np.argsort(similarities)[-k:]
                    top_k_relevance = true_relevance[top_k_indices]
                    precision_k = np.sum(top_k_relevance > 0) / k
                    ndcg_k = ndcg_at_k(top_k_relevance, similarities[top_k_indices], k)
                    evaluation_results.append({
                        'language': lang,
                        'query_id': query_id,
                        'precision_at_k': precision_k,
                        'ndcg_at_k': ndcg_k,
                        'k': k
                    })

    return pd.DataFrame(evaluation_results)

if __name__ == "__main__":
    # Load datasets
    datasets = load_datasets()
    data = load_data(datasets, lang='ru', sample_ratio=1.0, max_size=None) # specify based on comment in data_exploration.py

    # generate embeddings for queries and documents
    embeddings = sbert_embed_data(data)
    query_embeddings = embeddings['ru']['query_embeddings']
    doc_embeddings = embeddings['ru']['doc_embeddings']

    # perform retrieval to calculate similarities
    retrieval_results = {'ru': retrieve_documents(query_embeddings, doc_embeddings)}

    # prepare qrels for evaluation
    qrels = {lang: data[lang]['qrels'] for lang in data}

    # Evaluate retrieval results
    results_df = evaluate_retrieval(retrieval_results, qrels)
    print("\nEvaluation Results:")
    print(results_df)