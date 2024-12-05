import chromadb
import pandas as pd
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

from explore.funcs import load_datasets
import sys
import os

# Add parent folder to sys.path
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_folder)



# Initialize model
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="../db")
collection = chroma_client.get_or_create_collection(name="neuclir-titles-zh-bge-m3", metadata={"hnsw:space": "cosine"})
collection_sparse = chroma_client.get_or_create_collection(name="neuclir-titles-zh-bge-m3-sparse", metadata={"hnsw:space": "cosine"})


# Load datasets
datasets = load_datasets(["zh"])
dataset = datasets["zh"]
docs = pd.DataFrame(dataset.docs_iter())

# Preprocess documents
#preprocessor = TextPreprocessor("russian")
docs["title"] = docs["title"].fillna("")

# Split the data into batches
batches = [docs[i:i+100] for i in range(0, len(docs), 100)]

# Process and add documents to the collection
for i, batch in enumerate(tqdm(batches)):
    title_embeddings = model.encode(batch["title"].to_list(), return_dense=True, return_sparse=True, return_colbert_vecs=False)
    metadata = []
    for j in range(len(batch)):
        metadata.append({"url": batch["url"].iloc[j], "text": batch["text"].iloc[j]})

    collection.add(documents=batch["title"].to_list(), embeddings=title_embeddings["dense_vecs"], ids=batch["doc_id"].to_list(), metadatas=metadata)
    collection_sparse.add(documents=batch["title"].to_list(), embeddings=title_embeddings["sparse_vecs"], ids=batch["doc_id"].to_list(), metadatas=metadata)

