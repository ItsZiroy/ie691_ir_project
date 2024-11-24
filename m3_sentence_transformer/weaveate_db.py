
import os
import sys
#workaround to import modules from parent directory
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
import pickle

from FlagEmbedding import BGEM3FlagModel
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
from explore.funcs import load_datasets
import weaviate

import base64


def to_blob(obj):
    return base64.b64encode(pickle.dumps(obj)).decode('utf-8')

load_dotenv()

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

client = weaviate.connect_to_custom(http_host=os.getenv("WEAVIATE_HTTP_HOST"),http_port=int(os.getenv("WEAVIATE_HTTP_PORT")), http_secure=True, grpc_host=os.getenv("WEAVIATE_GRPC_HOST"), grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT")), grpc_secure=True, auth_credentials=weaviate.auth.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY")))

datasets = load_datasets(["zh", "ru", "fa"])
docs = pd.concat([pd.DataFrame(dataset.docs_iter()) for dataset in datasets.values()])
print(f"Running for: {len(docs)} docs.")

batches = [docs[i:i+10000] for i in range(0, len(docs), 10000)]
hc4 = client.collections.get("hc4_filtered_bge_m3")

outer_progress = tqdm(total=len(docs))

for i, batch in enumerate(batches):
    title_embeddings = model.encode(batch["title"].to_list(), return_dense=True, return_sparse=True, return_colbert_vecs=True)
    #doc_embeddings = model.encode(batch["text"].to_list(), return_dense=True, return_sparse=True, return_colbert_vecs=False)
    title_sparse_blobs = [to_blob(x) for x in title_embeddings["lexical_weights"]]
    title_colbert_blobs = [to_blob(x) for x in title_embeddings["colbert_vecs"]]
    batch = batch.reset_index(drop=True)
    with hc4.batch.fixed_size(batch_size=200, concurrent_requests=10) as b:
        for row in batch.itertuples(index=True):
            b.add_object(properties={
                "doc_id": row.doc_id,
                "title": row.title,
                "text": row.text,
                "url": row.url,
                "title_sparse": title_sparse_blobs[row.Index],
                "title_colbert": title_colbert_blobs[row.Index],
            }, vector={
                "title_dense": title_embeddings["dense_vecs"][row.Index],
            }, uuid=row.doc_id)
            outer_progress.update(1)