
import os
import sys
from time import sleep

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

import ir_datasets


def to_blob(obj):
    return base64.b64encode(pickle.dumps(obj)).decode('utf-8')

load_dotenv()

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

client = weaviate.connect_to_local()
dataset = ir_datasets.load("neuclir/1/multi/trec-2023")

print(f"Running for: {dataset.docs_count()} docs.")

batches = [(i, i+10000) for i in range(0, dataset.docs_count(), 10000)]
coll = client.collections.get("neuclir_1_mutli_bge_m3")

outer_progress = tqdm(total=dataset.docs_count(), initial=0)

for i, (start, end) in enumerate(batches):
    if i % 50 == 0 and i != 0:
        print(f"Sleeping for 5 minutes to allow indexing.")
        sleep(300)
    batch = pd.DataFrame([doc for doc in dataset.docs_iter()[start:end]])
    title_embeddings = model.encode(batch["title"].to_list(), return_dense=True, return_sparse=False,
                                    return_colbert_vecs=False)
    # doc_embeddings = model.encode(batch["text"].to_list(), return_dense=True, return_sparse=True, return_colbert_vecs=False)
    # title_sparse_blobs = [to_blob(x) for x in title_embeddings["lexical_weights"]]
    # title_colbert_blobs = [to_blob(x) for x in title_embeddings["colbert_vecs"]]
    batch = batch.reset_index(drop=True)
    with coll.batch.fixed_size(60, 2) as b:
        for row in batch.itertuples(index=True):
            b.add_object(properties={
                "doc_id": row.doc_id,
                "title": row.title,
                "text": row.text,
                "url": row.url
                # "title_sparse": title_sparse_blobs[row.Index],
                # "title_colbert": title_colbert_blobs[row.Index],
            }, vector={
                "title_dense": title_embeddings["dense_vecs"][row.Index],
            }, uuid=row.doc_id)
            outer_progress.update(1)
        if b.number_errors != 0:
            print(f"Found Errors: {b.number_errors}")

        b.flush()
        sleep(10)
