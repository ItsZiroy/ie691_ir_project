import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sbert_embedding import model_sbert

def retrieve_documents(query_embedding, doc_embeddings):
    """Calculate cosine similarity between the query embedding and document embeddings."""
    similarities = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings)
    return similarities.flatten()

def perform_retrieval(data, sbert_embeddings):
    """Perform document retrieval for each language based on the provided SBERT embeddings."""
    retrieval_results = {}
    
    for lang, lang_data in data.items():
        query_embeddings = sbert_embeddings[lang]['query_embeddings']
        doc_embeddings = sbert_embeddings[lang]['doc_embeddings']

        print(f"\nRetrieving documents for language: {lang.upper()}")

        retrieval_results[lang] = []
        for query_embedding in query_embeddings:
            similarities = retrieve_documents(query_embedding, doc_embeddings)
            retrieval_results[lang].append(similarities)

    return retrieval_results

def perform_custom_query_retrieval(custom_query, embeddings, lang, model, data):
    """Retrieve documents for a user-defined query using the specified model."""
    if model == 'sbert':
        query_embedding = model_sbert.encode(custom_query, show_progress_bar=False)
    else:
        raise ValueError(f"Model {model} is not supported.")

    results = {}
    doc_embeddings = embeddings[lang]['doc_embeddings']
    docs_df = data[lang]['docs'] 

    print(f"\nRetrieving documents for the custom query in language: {lang.upper()} using {model.upper()}")

    # calculate similarities
    similarities = retrieve_documents(query_embedding, doc_embeddings)
    # Get top N docs
    top_n_indices = np.argsort(similarities)[-5:][::-1] 
    results['similarities'] = similarities[top_n_indices]
    results['documents'] = docs_df.iloc[top_n_indices] 

    return results

if __name__ == "__main__":
    from data_preparation import load_datasets, load_data
    from sbert_embedding import sbert_embed_data
    
    #load data
    datasets = load_datasets()
    data = load_data(datasets, lang='ru', sample_ratio=1.0, max_size=None) # specify based on comment in data_exploration.py

    #compute embeddings
    sbert_embeddings = sbert_embed_data(data)
    retrieval_results = perform_retrieval(data, sbert_embeddings)

    # retrieval results
    for lang, results in retrieval_results.items():
        print(f"\nResults for language {lang}:")
        for i, similarities in enumerate(results):
            # average similarity
            average_similarity = np.mean(similarities)
            # index of highest similarity score
            best_index = np.argmax(similarities)
            best_similarity = similarities[best_index]

            print(f"\nQuery {i + 1}:")
            print(f"  Average Similarity: {average_similarity:.4f}")
            print(f"  Best Document Index: {best_index}, Best Similarity: {best_similarity:.4f}")

    # user-defined query
    custom_queries = {
        'sbert': input("Enter your custom English query for SBERT: "),
    }

    for model in custom_queries:
        target_language = 'ru'  # specify language
        custom_query = custom_queries[model]
        custom_results = perform_custom_query_retrieval(custom_query, sbert_embeddings, target_language, model, data)  # Pass `data` argument
        
        # custom query results
        print(f"\nCustom Query Results for language {target_language} using {model.upper()}:")
        for idx, doc in enumerate(custom_results['documents'].itertuples(), start=1):
            print(f"Document {idx}: {doc.text} (Similarity: {custom_results['similarities'][idx-1]:.4f})")