
# LMIR and LMIR-100K Retrieval Systems

## LMIR Retrieval System

### Requirements

Install the required dependencies:

``` bash
pip install numpy pandas nltk pymystem3 ir-datasets ir-measures
```

### Imports

``` bash
import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
import ir_datasets
import ir_measures
from ir_measures import nDCG, P, Judged, RBP, AP, RR, R
```

### Steps to Run the Project

#### Preprocess the Documents:

-   Load the dataset using ir_datasets (e.g., the Russian dataset
    \"ru\").
-   Fill NaN values in the text fields.
-   Tokenize, clean, and lemmatize the text using pymystem3:
-   Remove stop words and punctuation.
-   Convert text to lowercase.

#### Preprocess the Queries:

-   Apply consistent preprocessing (tokenization, stopword removal, and
    lemmatization) to the queries.

#### Create Unigram Language Models:

-   Construct unigram language models for each document and query.
-   Apply smoothing (e.g., $\alpha = 0.1$) to handle sparse data.

#### Calculate Similarity Scores Using KL Divergence:

-   Compute KL divergence between query and document models for ranking.
-   Rank documents based on their KL divergence scores for each query.

#### Output Results:

-   Save ranked results to a CSV file.
-   Evaluate the model using metrics such as <nDCG@20>, <P@5>, and
    <R@1000>.

-----
## LMIR-100K Retrieval System

### Requirements

Install the required dependencies:

``` bash
pip install numpy pandas nltk pymystem3 ir-datasets ir-measures
```

### Imports

``` bash
import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
import ir_datasets
import ir_measures
from ir_measures import nDCG, P, Judged, RBP, AP, RR, R
```

### Steps to Run the Project

#### Subset the Dataset to 100K Documents:

-   Filter documents with relevance judgments.
-   Randomly sample additional documents to reach a total of 100,000.
-   Save the subset as a new dataset.

#### Preprocess the Documents: 

-   Apply tokenization, lemmatization, and cleaning as done in LMIR.

#### Preprocess the Queries:

-   Use consistent preprocessing steps for queries.

#### Create Unigram Language Models:

-   Build unigram language models with smoothing for documents and
    queries in the 100K subset.

#### Calculate Similarity Scores Using KL Divergence:

-   Compute KL divergence between query and document models for ranking.
-   Evaluate the subset using the same metrics as LMIR.

#### Output Results:

-   Save ranked results to a CSV file.
-   Evaluate the model\'s performance on the reduced dataset.
:::
