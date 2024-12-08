import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the precomputed SBERT embeddings from saved file
def load_embeddings(save_path, lang):
    doc_embeddings_file = os.path.join(save_path, f"{lang}_doc_embeddings.npy")
    query_embeddings_file = os.path.join(save_path, f"{lang}_query_embeddings.npy")
    
    if not os.path.exists(doc_embeddings_file) or not os.path.exists(query_embeddings_file):
        raise FileNotFoundError(f"Embeddings files for {lang} not found in {save_path}.")
    
    doc_embeddings = np.load(doc_embeddings_file)
    query_embeddings = np.load(query_embeddings_file)

    return doc_embeddings, query_embeddings

# Function to calculate cosine similarity
def retrieve_documents(query_embedding, doc_embeddings):
    """Calculate cosine similarity between the query embedding and document embeddings."""
    similarities = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings)
    return similarities.flatten()

def perform_custom_query_retrieval(custom_query, embeddings, lang, data, save_path):
    """Retrieve documents for a user-defined query in English using precomputed embeddings."""
    
    # Load the document embeddings for the specified language (e.g., Russian, Multi)
    doc_embeddings = embeddings[lang]['doc_embeddings']
    docs_df = data[lang]['docs'] 
    
    # Initialize the model for encoding the custom English query
    model_sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    
    # Encode the custom English query
    query_embedding = model_sbert.encode(custom_query, show_progress_bar=False)
    
    # Perform document retrieval based on the custom query embedding
    print(f"\nRetrieving documents for the custom query in language: {lang.upper()}")

    # Calculate similarities between the custom query and all documents
    similarities = retrieve_documents(query_embedding, doc_embeddings)
    top_n_indices = np.argsort(similarities)[-10:][::-1]  # Get top N docs (sorted by highest similarity)

    # Prepare the results
    results = {
        'similarities': similarities[top_n_indices],
        'documents': docs_df.iloc[top_n_indices]
    }

    # Save the custom query retrieval results to CSV
    custom_query_results_file = os.path.join(save_path, f"{lang}_custom_query_results.csv")
    custom_query_df = pd.DataFrame({
        'document_index': top_n_indices,
        'similarity': results['similarities'],
        'document_text': results['documents']['text']
    })
    custom_query_df.to_csv(custom_query_results_file, index=False)
    print(f"Custom query results saved to {custom_query_results_file}")

    return results

# Main function to load embeddings and perform retrieval
if __name__ == "__main__":
    # Path to saved embeddings and data
    save_path = './sbert_sentence_transformer/output'
    
    # Load precomputed embeddings (Russian language example)
    lang = 'ru'
    doc_embeddings, query_embeddings = load_embeddings(save_path, lang)

    # Embeddings dictionary format for easy retrieval
    embeddings = {
        lang: {
            'doc_embeddings': doc_embeddings,
            'query_embeddings': query_embeddings
        }
    }

    # Load the corresponding documents and queries
    docs_df = pd.read_csv(os.path.join(save_path, f'{lang}_filtered_docs.csv'))
    queries_df = pd.read_csv(os.path.join(save_path, f'{lang}_queries.csv'))

    # Package the data for easier access
    data = {
        lang: {
            'docs': docs_df,
            'queries': queries_df.to_dict(orient='records')
        }
    }

    # Get custom query input (in English)
    custom_query = input("Enter your custom English query: ")

    # Perform retrieval for the custom query in English
    results = perform_custom_query_retrieval(custom_query, embeddings, lang, data, save_path)

    # Display the results
    print(f"\nCustom Query Results for language {lang.upper()}:")
    for idx, doc in enumerate(results['documents'].itertuples(), start=1):
        print(f"Document {idx}: {doc.text} (Similarity: {results['similarities'][idx-1]:.4f})")