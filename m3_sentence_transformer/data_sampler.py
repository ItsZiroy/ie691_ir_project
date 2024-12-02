import math
import random

import pandas as pd

from explore.funcs import load_datasets
import ir_datasets


def get_sample_docs_with_all_qrels(path: str):
    d = pd.read_csv(path, encoding="utf-8", dtype={"doc_id": str, "title": str, "text": str, "url": str})
    d = d[d["title"].apply(lambda x: isinstance(x, str))]
    d = d.sample(frac=1, random_state=42)
    print(f"Loaded {len(d)} docs.")
    return d


class DataSampler:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = ir_datasets.load(dataset_name)
        self.queries = pd.DataFrame(self.dataset.queries_iter())
        self.qrels = pd.DataFrame(self.dataset.qrels_iter())

        self.docstore = self.dataset.docs_store()

    def valid_qrels(self) -> pd.DataFrame:
        """
        Filter qrels to only include those with matching documents and queries.

        Returns:
            pd.DataFrame: Filtered qrels.
        """
        return self.qrels

    def valid_queries(self) -> pd.DataFrame:
        """
        Filter queries to only include those with matching documents and queries.

        Returns:
            pd.DataFrame: Filtered queries.
        """
        valid_query_ids = set(self.qrels["query_id"])
        return self.queries[self.queries["query_id"].isin(valid_query_ids)]

    def sample_queries(self, n) -> pd.DataFrame:
        """
        Sample n queries from the dataset.

        Args:
            n (int): Number of queries to sample.

        Returns:
            pd.DataFrame: Sampled queries.
        """
        valid_queries = self.valid_queries()
        sample_query = valid_queries.sample(n=n, random_state=42)
        qrels = self.qrels[self.qrels["query_id"].isin(sample_query["query_id"])]

        combined = qrels.merge(sample_query, left_on="query_id", right_on="query_id", how="left")
        return combined

    def sample(self, relevance_scores, min_sample_count) -> pd.DataFrame:
        """
        Sample equally from each relevance score based on the minimum count and match documents and queries.

        Args:
            relevance_scores (list): List of relevance scores to sample from.
            min_sample_count (int): Minimum number of samples to retrieve for each relevance score.

        Returns:
            dict: A dictionary with matched documents and queries.
        """
        samples = []
        for score in relevance_scores:
            sample = self.qrels[self.qrels["relevance"] == score].sample(n=min_sample_count, random_state=42)
            samples.append(sample)

        # Combine sampled queries and documents
        final_sample = pd.concat(samples).reset_index(drop=True)

        # Extract IDs in the exact order they appear in final_sample
        doc_ids = final_sample["doc_id"].values
        query_ids = final_sample["query_id"].values

        print(query_ids)

        # Match docs and queries, preserving duplicates and order
        matched_docs = self.docs.set_index("doc_id").reindex(doc_ids).reset_index()
        matched_queries = self.queries.set_index("query_id").reindex(query_ids).reset_index()

        # Combine matched_queries and matched_docs into one DataFrame for the same order
        combined = final_sample.merge(
            matched_queries, left_on="query_id", right_on="query_id", how="left"
        ).merge(
            matched_docs, left_on="doc_id", right_on="doc_id", how="left"
        )

        return combined

    def valid_docs(self):
        """
        Filter documents to only include those with matching documents and queries.

        Returns:
            pd.DataFrame: Filtered documents.
        """
        valid_doc_ids = set(self.qrels["doc_id"])
        return self.docstore.get_many(valid_doc_ids)

    def create_sample_docs_with_all_qrels(self, num_samples):
        arr = []
        for doc in self.docstore.get_many_iter(set(self.qrels["doc_id"])):
            arr.append({
                "doc_id": doc.doc_id,
                "title": doc.title,
                "text": doc.text,
                "url": doc.url
            })
        df2 = pd.DataFrame(arr)
        print(len(df2))

        if num_samples - len(df2) <= 0:
            raise Exception("Not enough samples selected")

        splitter = math.floor(self.dataset.docs_count() / (num_samples - len(df2)))


        random_docs = self.dataset.docs_iter()[::splitter]
        df = pd.DataFrame(random_docs)[["doc_id", "title", "text", "url"]]
        print(len(df))


        merged = df._append(df2)
        return merged.drop_duplicates()


if __name__ == "__main__":
    sampler = DataSampler("neuclir/1/multi/trec-2023")
    df = sampler.create_sample_docs_with_all_qrels(1000000)

    df.to_csv("random_docs_with_qrels_1m.csv", index=False, encoding="utf-8")
    print(len(df))

