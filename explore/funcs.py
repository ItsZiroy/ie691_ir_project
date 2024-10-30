import pandas as pd

def calculate_average_document_length(datasets):
    """
    Calculate the average document length for each dataset in the given dictionary.
    
    Args:
        datasets (dict): A dictionary containing dataset variable names as keys 
                         and the datasets as values.
    
    Returns:
        dict: A dictionary with dataset variable names as keys and their average 
              document lengths as values.
    """
    avg_lengths = {}  # dict. to store average lengths
    for var_name, dataset in datasets.items():
        documents = [doc for doc in dataset.docs_iter()]
        df = pd.DataFrame(documents)
        
        # calculate average length of the document
        avg_length = df['text'].str.len().mean()  # Adjust based on the actual column name
        avg_lengths[var_name] = avg_length
        print(f"Average document length in {var_name}: {avg_length:.2f} characters")
    
    return avg_lengths  # return the dict. of average lengths

import pandas as pd

def analyze_query(datasets, query=None):
    """
    Analyze the datasets by calculating average document length and counting documents containing a specific query.
    
    Args:
        datasets (dict): A dictionary containing dataset variable names as keys 
                         and the datasets as values.
        query (str): A keyword to search for in the documents (default: None).
    
    Returns:
        dict: A dictionary containing average document lengths and document counts for the given query.
    """
    results = {
        "average_lengths": {},
        "query_counts": {}
    }
    
    for var_name, dataset in datasets.items():
        documents = [doc for doc in dataset.docs_iter()]
        df = pd.DataFrame(documents)
        
        # calculate average length of the document
        avg_length = df['text'].str.len().mean()  # Adjust based on the actual column name
        results["average_lengths"][var_name] = avg_length
        
        # if query provided, count matching documents
        if query:
            matching_docs = df[df['text'].str.contains(query, na=False)]  # Adjust based on actual column name
            results["query_counts"][var_name] = len(matching_docs)
    
    return results