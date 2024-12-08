import ir_datasets
import pandas as pd
import os

def load_datasets():
    datasets = {
        "ru": ir_datasets.load("neuclir/1/multi/trec-2023"),
        "zh": ir_datasets.load("neuclir/1/zh/hc4-filtered"),
        "zh": ir_datasets.load("neuclir/1/zh/hc4-filtered"),
        "multi": ir_datasets.load("neuclir/1/multi/trec-2023")
        
    }
    return datasets

def load_data(datasets, lang=None, max_size=None, save_path=None):
    data = {}
    languages = [lang] if lang else datasets.keys()

    for lang in languages:
        dataset = datasets[lang]
        print(f"\nLoading data for language: {lang.upper()}")

        # Load qrels and get relevant doc_ids
        qrels_iter = dataset.qrels_iter()
        qrels = [{'query_id': qrel.query_id, 'doc_id': qrel.doc_id, 'relevance': qrel.relevance} for qrel in qrels_iter]
        qrels_df = pd.DataFrame(qrels)
        relevant_doc_ids = set(qrels_df['doc_id'])  # Extract relevant doc_ids

        # Filter only relevant documents
        docs_iter = dataset.docs_iter()
        filtered_docs = [doc for doc in docs_iter if doc.doc_id in relevant_doc_ids]  # Keep all attributes

        # Load queries
        queries_iter = dataset.queries_iter()
        queries = list(queries_iter)  # Keep all attributes

        # Convert qrels to DataFrame
        qrels_df = pd.DataFrame(qrels) 

        # Save filtered docs, queries, and qrels to CSV
        if save_path:
            os.makedirs(save_path, exist_ok=True)

            # Save filtered docs
            docs_file = os.path.join(save_path, f"{lang}_filtered_docs.csv")
            docs_df = pd.DataFrame([doc._asdict() for doc in filtered_docs])  # Convert namedtuple to dict
            docs_df.to_csv(docs_file, index=False)

            # Save queries
            queries_file = os.path.join(save_path, f"{lang}_queries.csv")
            queries_df = pd.DataFrame([query._asdict() for query in queries]) 
            queries_df.to_csv(queries_file, index=False)

            # Save qrels
            qrels_file = os.path.join(save_path, f"{lang}_filtered_qrels.csv")
            qrels_df.to_csv(qrels_file, index=False)

            print(f"Filtered documents saved to {docs_file}.")
            print(f"Filtered queries saved to {queries_file}.")  
            print(f"Filtered qrels saved to {qrels_file}.")

        data[lang] = {
            'docs': filtered_docs, 
            'queries': queries,    
            'qrels': qrels_df      
        }

        print(f"Loaded {len(filtered_docs)} relevant documents, {len(queries)} queries, and {len(qrels_df)} qrels for language {lang.upper()}.")

    return data

if __name__ == "__main__":
    datasets = load_datasets()
    data = load_data(datasets, lang='ru', max_size=None, save_path="./sbert_sentence_transformer/output")