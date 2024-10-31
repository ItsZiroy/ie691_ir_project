import pandas as pd
import ir_datasets

# mapping dataset to variable names
dataset_vars = {
    "neuclir/1/fa": "fa",
    "neuclir/1/fa/hc4-filtered": "fa_hc4_filtered",
    "neuclir/1/fa/trec-2022": "fa_trec_2022",
    "neuclir/1/fa/trec-2023": "fa_trec_2023",
    "neuclir/1/multi": "multi",
    "neuclir/1/multi/trec-2023": "multi_trec_2023",
    "neuclir/1/ru": "ru",
    "neuclir/1/ru/hc4-filtered": "ru_hc4_filtered",
    "neuclir/1/ru/trec-2022": "ru_trec_2022",
    "neuclir/1/ru/trec-2023": "ru_trec_2023",
    "neuclir/1/zh": "zh",
    "neuclir/1/zh/hc4-filtered": "zh_hc4_filtered",
    "neuclir/1/zh/trec-2022": "zh_trec_2022",
    "neuclir/1/zh/trec-2023": "zh_trec_2023"
}

### DOC & QUERY COUNTS ###
# loading each dataset
loaded_datasets = {}
for ds_id, var_name in dataset_vars.items():
    try:
        loaded_datasets[var_name] = ir_datasets.load(ds_id)
        print(f"Dataset: {var_name}")
        # print each dataset's document and query counts
        if hasattr(loaded_datasets[var_name], 'docs_count'):
            print(f"  Number of documents: {loaded_datasets[var_name].docs_count()}")
        if hasattr(loaded_datasets[var_name], 'queries_count'):
            print(f"  Number of queries: {loaded_datasets[var_name].queries_count()}")
        if hasattr(loaded_datasets[var_name], 'qrels'):
            print(f"  Number of qrels: {loaded_datasets[var_name].qrels}")
    except Exception as e:
        print(f"Error loading dataset {var_name}: {e}")
    print("=" * 40)

### EXPLORE PERSIAN DOC ###
# load persian datasets
persian_datasets = {
    "fa": loaded_datasets["fa"],
    #"fa_hc4_filtered": loaded_datasets["fa_hc4_filtered"],
    #"fa_trec_2022": loaded_datasets["fa_trec_2022"],
    #"fa_trec_2023": loaded_datasets["fa_trec_2023"]
}

# display head of selected Persian dataset
for var_name, dataset in persian_datasets.items():
    print(f"First document from dataset: {var_name}")
    documents = [doc for doc in dataset.docs_iter()[:10]]
    df = pd.DataFrame(documents)
    print(df.head())
    print("=" * 40)

"""### EXPLORE RUSSIAN DOC ###
# load Russian datasets
russian_datasets = {
    "ru": loaded_datasets["ru"],
    "ru_hc4_filtered": loaded_datasets["ru_hc4_filtered"],
    "ru_trec_2022": loaded_datasets["ru_trec_2022"],
    "ru_trec_2023": loaded_datasets["ru_trec_2023"]
}

# display head of selected Russian dataset
for var_name, dataset in russian_datasets.items():
    print(f"First document from dataset: {var_name}")
    documents = [doc for doc in dataset.docs_iter()]
    df = pd.DataFrame(documents)
    print(df.head(1))
    print("=" * 40)

### EXPLORE CHINESE DOC ###
# load Chinese datasets
chinese_datasets = {
    "zh": loaded_datasets["zh"],
    "zh_hc4_filtered": loaded_datasets["zh_hc4_filtered"],
    "zh_trec_2022": loaded_datasets["zh_trec_2022"],
    "zh_trec_2023": loaded_datasets["zh_trec_2023"]
}

# display head of selected Chinese dataset
for var_name, dataset in chinese_datasets.items():
    print(f"First document from dataset: {var_name}")
    documents = [doc for doc in dataset.docs_iter()]
    df = pd.DataFrame(documents)
    print(df.head(1))
    print("=" * 40)

### EXPLORE MULTI DOC ###
# load multi datasets
multi_datasets = {
    "multi": loaded_datasets["multi"],
    "multi_trec_2023": loaded_datasets["multi_trec_2023"]
}

# display head of selected multi-language dataset
for var_name, dataset in multi_datasets.items():
    print(f"First document from dataset: {var_name}")
    documents = [doc for doc in dataset.docs_iter()]
    df = pd.DataFrame(documents)
    print(df.head(1))
    print("=" * 40)"""