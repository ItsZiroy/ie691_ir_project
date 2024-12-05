### Setup
Make sure to have downloaded the samples as described in the main [README.md](../README.md#samples).

To run these experiments, you must have a local weaviate instance running.
For this, you may use the docker compose in the `weaviate` folder in the root
of the project. Recommended minimum RAM is 16GB as well as all cores that are 
available to you.

Start the container with the following command:

```bash
docker-compose up -d
```
> [!NOTE]
> All mentioned script are located in `m3_sentence_transformer` folder.

After the container is running, you can start the upload using the `weaviate_setup.ipynb` jupyter notebook.
Make sure to update the following lines to match the sample you would like to use
and the collection name you would like to create:

```python
sample_path = "neuclir_1_mutli_bge_m3_100k.csv"
collection_name = "neuclir_1_mutli_bge_m3_100k"
collbert_collection_name = "neuclir_1_mutli_bge_m3_100k_colbert"
```

> [!TIP]
> In my experience you should create a fresh instance of weaviate for each sample set you would like to upload.
> You can do so by altering the storage file path in the `weaviate/docker-compose.yml` file.

### Inserting Embeddings into Weaviate

Modify the following lines of the `weaviate_db_sample.py` script to match your setup:

File path to the sample csv:
```python
docs = get_sample_docs_with_all_qrels("random_docs_with_qrels_200k.csv")
```

Name of the created collection from the above setp:
```python
coll = client.collections.get("neuclir_1_mutli_bge_m3_200k")
```

Then run the script:

```bash
poetry run python weaviate_db_sample.py
```
The inserts will take a while, depending on the size of the dataset.

### Insert Colbert Embeddings into Weaviate

The script can be found in `weaviate_db_sample_colbert.py`.
You have to modify the same lines as [above](#insert-colbert-embeddings-into-weaviate) and additonally edit the
following line to match the collection you created for colbert:

```python
coll_colbert = client.collections.get("neuclir_1_mutli_bge_m3_100k_colbert")
```

Then run the script:

```bash
poetry run python weaviate_db_sample_colbert.py
```

### Evaluation

To evaluate the runs, simply run the `evaluation_weaviate.ipynb` jupyter notebook:

### Misc
All other files and experiments that lead me to the final setup can be found in the `misc` folder.
