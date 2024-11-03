import ir_datasets
import pandas as pd

def load_datasets():
    datasets = {
        "ru": ir_datasets.load("neuclir/1/ru/hc4-filtered"),
        "zh": ir_datasets.load("neuclir/1/zh/hc4-filtered"),
        "fa": ir_datasets.load("neuclir/1/fa/hc4-filtered")
    }
    return datasets

def load_data(datasets, lang=None, sample_ratio=1.0, max_size=None):
    data = {}
    languages = [lang] if lang else datasets.keys()
    
    for lang in languages:
        dataset = datasets[lang]
        print(f"\nLoading data for language: {lang.upper()}")

        docs = []
        queries = []
        qrels = []
        
        docs_iter = dataset.docs_iter()
        queries_iter = dataset.queries_iter()
        qrels_iter = dataset.qrels_iter()
        
        # load docs
        for i, doc in enumerate(docs_iter):
            if max_size and i >= max_size:
                break
            docs.append(doc)

        # load queries
        for i, query in enumerate(queries_iter):
            if max_size and i >= max_size:
                break
            queries.append(query)

        # load qrels
        for i, qrel in enumerate(qrels_iter):
            if max_size and i >= max_size:
                break
            qrels.append(qrel)

        # convert to DF
        docs_df = pd.DataFrame(docs)
        queries_df = pd.DataFrame(queries)
        qrels_df = pd.DataFrame(qrels)

        # balanced sampling based on relevance scores in qrels
        if sample_ratio < 1.0:
            relevance_counts = qrels_df['relevance'].value_counts(normalize=True)
            sample_sizes = (relevance_counts * sample_ratio * len(qrels_df)).round().astype(int)
            sampled_qrels = []
            for rel_level, count in sample_sizes.items():
                if count > 0:
                    sampled_qrels.extend(qrels_df[qrels_df['relevance'] == rel_level].sample(n=count, random_state=42).values.tolist())
            sampled_qrels_df = pd.DataFrame(sampled_qrels, columns=qrels_df.columns)
            qrels_df = sampled_qrels_df

        data[lang] = {
            'docs': docs_df,
            'queries': queries_df,
            'qrels': qrels_df
        }

        print(f"Loaded {len(docs_df)} documents, {len(queries_df)} queries, and {len(qrels_df)} qrels for language {lang.upper()}.")
        print(f"Sample documents ({lang}):\n", docs_df.head(), "\n")
        print(f"Sample queries ({lang}):\n", queries_df.head(), "\n")
        print(f"Sample qrels ({lang}):\n", qrels_df.head(), "\n")

    return data

if __name__ == "__main__":
    datasets = load_datasets()
    data = load_data(datasets, lang='ru', sample_ratio=1.0, max_size=None)  # specify sample_ratio 1.0 for full or i.e. 0.10 for 10% (load_data), max_size=None if whole "ru" dataset, otherwise specify number of entries (load_datasets)