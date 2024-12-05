from time import sleep
import pickle


import os
import sys
from time import sleep
#workaround to import modules from parent directory
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))

from data_sampler import get_sample_docs_with_all_qrels

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

client = weaviate.connect_to_local()

docs = get_sample_docs_with_all_qrels("random_docs_with_qrels_100k.csv")

batches = [(i, i + 10000) for i in range(0, len(docs), 10000)] # 600000
coll = client.collections.get("neuclir_1_mutli_bge_m3_100k")

outer_progress = tqdm(total=len(docs), initial=0)


for i, (start, end) in enumerate(batches):
    batch = docs[start:end]
    title_embeddings = model.encode(batch["title"].to_list(), return_dense=True, return_sparse=False,
                                    return_colbert_vecs=False)
    # doc_embeddings = model.encode(batch["text"].to_list(), return_dense=True, return_sparse=True, return_colbert_vecs=False)
    # title_sparse_blobs = [to_blob(x) for x in title_embeddings["lexical_weights"]]
    # title_colbert_blobs = [to_blob(x) for x in title_embeddings["colbert_vecs"]]
    batch = batch.reset_index(drop=True)
    with coll.batch.fixed_size(60, 2) as b:
        for row in batch.itertuples(index=True):
            # print(row)
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

