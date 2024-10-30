import pandas as pd
import ir_datasets
from Zain.explore.funcs import calculate_average_document_length 

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

# loading each dataset
loaded_datasets = {}
for ds_id, var_name in dataset_vars.items():
    try:
        loaded_datasets[var_name] = ir_datasets.load(ds_id)
        print(f"Dataset: {var_name} loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset {var_name}: {e}")

# defining and populating datasets for different languages
persian_datasets = {
    "fa": loaded_datasets.get("fa"),
    #"fa_hc4_filtered": loaded_datasets.get("fa_hc4_filtered"),
    #"fa_trec_2022": loaded_datasets.get("fa_trec_2022"),
    #"fa_trec_2023": loaded_datasets.get("fa_trec_2023")
}

russian_datasets = {
    "ru": loaded_datasets.get("ru"),
    "ru_hc4_filtered": loaded_datasets.get("ru_hc4_filtered"),
    "ru_trec_2022": loaded_datasets.get("ru_trec_2022"),
    "ru_trec_2023": loaded_datasets.get("ru_trec_2023")
}

chinese_datasets = {
    "zh": loaded_datasets.get("zh"),
    "zh_hc4_filtered": loaded_datasets.get("zh_hc4_filtered"),
    "zh_trec_2022": loaded_datasets.get("zh_trec_2022"),
    "zh_trec_2023": loaded_datasets.get("zh_trec_2023")
}

multi_datasets = {
    "multi": loaded_datasets.get("multi"),
    "multi_trec_2023": loaded_datasets.get("multi_trec_2023")
}

# Calculate and print average document lengths for Persian datasets
avg_lengths_persian = calculate_average_document_length(persian_datasets)
print("Average Document Lengths (Persian):")
for dataset_name, avg_length in avg_lengths_persian.items():
    print(f"{dataset_name}: {avg_length:.2f} characters")
"""
# Calculate and print average document lengths for Russian datasets
avg_lengths_russian = calculate_average_document_length(russian_datasets)
print("Average Document Lengths (Russian):")
for dataset_name, avg_length in avg_lengths_russian.items():
    print(f"{dataset_name}: {avg_length:.2f} characters")

# Calculate and print average document lengths for Chinese datasets
avg_lengths_chinese = calculate_average_document_length(chinese_datasets)
print("Average Document Lengths (Chinese):")
for dataset_name, avg_length in avg_lengths_chinese.items():
    print(f"{dataset_name}: {avg_length:.2f} characters")

# Calculate and print average document lengths for Multi-language datasets
avg_lengths_multi = calculate_average_document_length(multi_datasets)
print("Average Document Lengths (Multi-language):")
for dataset_name, avg_length in avg_lengths_multi.items():
    print(f"{dataset_name}: {avg_length:.2f} characters")
    """