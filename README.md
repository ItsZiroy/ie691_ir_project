# IE691 Information Retrieval Project
## Installing Dependencies
This project uses Poetry to manage dependencies. To install it, 
you can use the [installation guide](https://python-poetry.org/docs/#installation) on their website.
To install the dependencies, run the following command:
```bash
poetry install
```

### Troubleshooting Python Kernel

If your code editor is having trouble finding the poetry kernel, you can locate it via
and manually set it in your code editor.

```bash
poetry env info --executable
```

Then, you can set the kernel in your code editor to the path of the virtual environment.

## Samples

Most of our evaluation runs are on a subset of the full dataset.
You can either generate them on your own using the script provided in
`m3_sentence_transformer/data_sampler.py` or download them from the following link:

https://1drv.ms/u/c/bbf8be11fdee265a/EWC4kI_vdTBEnPsFgskGuv0Bw4ZcDSaH-oyOqXczgvN6pA?e=qdt7sj

> [!NOTE]
> Depending on the run, you may have to change the path to the samples in the respective scripts.

## Reproducing examples

To reproduce the results use the dedicated README.md in the respective folder.

- [BGE M3 | ColBERT](m3_sentence_transformer/README.md)
- [SBERT](sbert_sentence_transformer/README.md)
- [LMIR | LMIR-100k](baselines/Readme_LMIR,LMIR-100k.md)
