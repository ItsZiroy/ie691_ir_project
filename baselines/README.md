# TF-IDF Retrieval Implementation
This project demonstrates how to implement a TF-IDF-based retrieval system to process queries and rank documents by relevance. The implementation is Python-based and makes use of commonly used libraries for natural language processing and information retrieval.

# Requirements
pip install numpy pandas scikit-learn nltk

# Imports
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from funcs import load_datasets, get_docs
from ir_measures 

# Steps to Run the Project

## Preprocess the Documents:
Load your dataset containing documents.
Tokenize and clean text by removing stop words, punctuation, and converting to lowercase.
Use nltk for preprocessing.

### Preprocess the Queries:
Prepare queries using the same preprocessing steps as for the documents.
Ensure tokenization and stop word removal are consistent.


## Create TF-IDF Representations:
Use TfidfVectorizer from sklearn to create TF-IDF matrices for the documents and queries.

## Compute Similarity Scores:
Use cosine_similarity to compute similarity scores between queries and documents.

## Rank Documents for Each Query:
For each query, rank documents based on similarity scores.
Normalize the scores (optional) for better interpretability.

## Output Results:
Create a ranking of documents for each query.

# BM25 Retrieval Implementation
BM25 is a ranking function in information retrieval, used to rank documents by relevance to a query.

## Requirements
pip install numpy pandas rank-bm25 nltk

## Imports
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Steps to Run the Project

## Preprocess the Documents:
Load your dataset containing documents.
Tokenize and clean text using NLTK:
Convert text to lowercase.
Remove stop words and punctuation.
Optionally, stem or lemmatize the tokens.

## Preprocess the Queries:
Preprocess queries in the same way as documents to ensure consistency.

## Initialize the BM25 Model:
Tokenize the preprocessed documents.
Create a BM25Okapi object using the tokenized documents.

## Retrieve Documents:
For each preprocessed query, use BM25Okapi.get_scores to compute scores for all documents.
Use BM25Okapi.get_top_n to retrieve the top 
𝑁
N most relevant documents for each query.

## Rank Documents:
Sort documents by their BM25 scores in descending order.
Normalize the scores (optional) to a range of [0, 1] for better interpretability.

## Output Results:
Create a ranking table with the query ID, document ID, score, and normalized score.