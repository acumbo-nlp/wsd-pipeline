# WSD Pipeline

This project compares four methods for choosing the meaning of *bank* in
English. The current experiment covers two senses, river bank and financial
institution.

## Results

Each dataset has three pairs of sentences. A pair counts as correct only
when the method predicts both meanings correctly.

| Method | Dev sentences | Dev pairs | Test sentences | Test pairs |
|---|---:|---:|---:|---:|
| Keywords | 4/6 | 1/3 | 3/6 | 0/3 |
| Lesk | 5/6 | 2/3 | 3/6 | 0/3 |
| MiniLM, full sentence | 4/6 | 1/3 | 4/6 | 1/3 |
| MiniLM, four words on either side | 5/6 | 2/3 | 2/6 | 0/3 |

Full-sentence MiniLM was the only method to solve `late_clue`. The two
sentences in that pair have identical local windows, and the words that
identify the correct meanings appear later.

## Run the test

With a Python 3.13 virtual environment active, run these commands.

```bash
python -m pip install -r requirements.txt
python -m nltk.downloader wordnet
python demo.py --method reranker --data data/test_pairs.csv
```

FastEmbed downloads the MiniLM model on the first run.