import os
import time
import pandas as pd
from data_preparation import load_datasets, load_data 
from sbert_embedding import sbert_embed_data
from retrieval import perform_custom_query_retrieval 

def main():
    # load datasets
    selected_lang = 'ru'  # select language
    datasets = load_datasets()
    data = load_data(datasets, lang=selected_lang, sample_ratio=1.0, max_size=None)
    print({"ru": "Russian dataset loaded.", 
           "zh": "Chinese dataset loaded.",
           "fa": "Persian dataset loaded."}.get(selected_lang, "Dataset loaded for the selected language."))

    # generate embeddings 
    start_time = time.time()
    embeddings = sbert_embed_data(data) 
    sbert_time = time.time() - start_time
    print(f"SBERT embeddings for {selected_lang} dataset computed in {sbert_time:.2f} seconds.")

    # get a custom query for retrieval
    custom_query = input("Enter a custom query for retrieval: ")

    # perform retrieval for custom query
    start_time = time.time()
    custom_results = perform_custom_query_retrieval(custom_query, embeddings, lang=selected_lang, model='sbert', data=data)  
    retrieval_time = time.time() - start_time
    print(f"Retrieval for custom query completed in {retrieval_time:.2f} seconds.")

    # save the retrieved documents to a file
    retrieved_docs_df = pd.DataFrame(custom_results['documents'])  
    
    # construct output path
    output_directory = r".\T1_sentence_transformers\Results"
    os.makedirs(output_directory, exist_ok=True) 
    output_filename = f"bert_retrieved_docs_{selected_lang}.csv" 
    output_path = os.path.join(output_directory, output_filename) 

    retrieved_docs_df.to_csv(output_path, index=False) 
    print(f"Retrieved documents for custom query saved to {output_path}.")

    # display similarities
    print("\nTop Similarities for the custom query:")
    for idx, similarity in enumerate(custom_results['similarities']):
        print(f"Document {idx + 1}: Similarity = {similarity:.4f}")

if __name__ == "__main__":
    main()