import ir_datasets
from Zain.explore.funcs import analyze_query 

# dataset mapping
dataset_vars = {
    "neuclir/1/fa": "fa",
    "neuclir/1/fa/hc4-filtered": "fa_hc4_filtered",
    "neuclir/1/fa/trec-2022": "fa_trec_2022",
    "neuclir/1/fa/trec-2023": "fa_trec_2023",
    "neuclir/1/ru": "ru",
    "neuclir/1/ru/hc4-filtered": "ru_hc4_filtered",
    "neuclir/1/ru/trec-2022": "ru_trec_2022",
    "neuclir/1/ru/trec-2023": "ru_trec_2023",
    "neuclir/1/zh": "zh",
    "neuclir/1/zh/hc4-filtered": "zh_hc4_filtered",
    "neuclir/1/zh/trec-2022": "zh_trec_2022",
    "neuclir/1/zh/trec-2023": "zh_trec_2023",
    "neuclir/1/multi": "multi",
    "neuclir/1/multi/trec-2023": "multi_trec_2023"
}

# loading datasets
loaded_datasets = {}
for ds_id, var_name in dataset_vars.items():
    try:
        loaded_datasets[var_name] = ir_datasets.load(ds_id)
        print(f"Dataset loaded: {var_name}")
    except Exception as e:
        print(f"Failed to load dataset {var_name}: {e}")

# defining Russian datasets
persian_datasets = {
    "fa": loaded_datasets.get("fa"),
    #"fa_hc4_filtered": loaded_datasets.get("fa_hc4_filtered"),
    #"fa_trec_2022": loaded_datasets.get("fa_trec_2022"),
    #"fa_trec_2023": loaded_datasets.get("fa_trec_2023")
}
# define keyword/query for Persian
persian_query = "تکنولوژی"

# analyze Persian datasets
if persian_datasets:
    print(f"\nAnalyzing Persian datasets with keyword '{persian_query}'...")
    results = analyze_query(persian_datasets, persian_query)
    
    # print query counts
    print(f"Documents containing '{persian_query}' (Persian):")
    for dataset_name, count in results["query_counts"].items():
        print(f"{dataset_name}: {count} documents")
"""
# define Russian datasets
russian_datasets = {
    "ru": loaded_datasets.get("ru"),
    "ru_hc4_filtered": loaded_datasets.get("ru_hc4_filtered"),
    "ru_trec_2022": loaded_datasets.get("ru_trec_2022"),
    "ru_trec_2023": loaded_datasets.get("ru_trec_2023")
}

# define keyword/query for Russian
russian_query = "технология" 

# analyze Russian datasets
if russian_datasets:
    print(f"\nAnalyzing Russian datasets with keyword '{russian_query}'...")
    results = analyze_query(russian_datasets, russian_query)
    
    # print query counts
    print(f"Documents containing '{russian_query}' (Russian):")
    for dataset_name, count in results["query_counts"].items():
        print(f"{dataset_name}: {count} documents")

# defining Chinese datasets
chinese_datasets = {
    "zh": loaded_datasets.get("zh"),
    "zh_hc4_filtered": loaded_datasets.get("zh_hc4_filtered"),
    "zh_trec_2022": loaded_datasets.get("zh_trec_2022"),
    "zh_trec_2023": loaded_datasets.get("zh_trec_2023")
}

# define keyword/query for Chinese
chinese_query = "技术"

# analyze Chinese datasets
if chinese_datasets:
    print(f"\nAnalyzing Chinese datasets with keyword '{chinese_query}'...")
    results = analyze_query(chinese_datasets, chinese_query)
    
    # print query counts
    print(f"Documents containing '{chinese_query}' (Chinese):")
    for dataset_name, count in results["query_counts"].items():
        print(f"{dataset_name}: {count} documents")

# defining the multi-language datasets
multi_datasets = {
    "multi": loaded_datasets.get("multi"),
    "multi_trec_2023": loaded_datasets.get("multi_trec_2023")
}

# define keyword/query for multi-language
multi_query = "technology" 

# analyze multi-language datasets
if multi_datasets:
    print(f"\nAnalyzing Multi-language datasets with keyword '{multi_query}'...")
    results = analyze_query(multi_datasets, multi_query)
    
    # print query counts
    print(f"Documents containing '{multi_query}' (Multi-language):")
    for dataset_name, count in results["query_counts"].items():
        print(f"{dataset_name}: {count} documents")
        """