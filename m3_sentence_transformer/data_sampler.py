import pandas as pd

from explore.funcs import load_datasets

class DataSampler:
    def __init__(self, language):
        self.language = language
        self.datasets = load_datasets([language])
        self.dataset = self.datasets[language]
        self.docs = pd.DataFrame(self.dataset.docs_iter())
        self.queries = pd.DataFrame(self.dataset.queries_iter())
        self.qrels = pd.DataFrame(self.dataset.qrels_iter())

        # Filter qrels to only include those with matching documents and queries
        valid_doc_ids = set(self.docs["doc_id"])
        valid_query_ids = set(self.queries["query_id"])
        self.qrels = self.qrels[self.qrels["doc_id"].isin(valid_doc_ids)]
        self.qrels = self.qrels[self.qrels["query_id"].isin(valid_query_ids)]

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
