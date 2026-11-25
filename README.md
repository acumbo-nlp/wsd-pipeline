# WSD Pipeline

How much context does a model need to choose the right meaning of an
ambiguous word? This project compares a pretrained MiniLM model reading a
full sentence with the same model reading up to four words on either side
of the target. Both versions rank definitions from WordNet 3.0.

## Controlled context test

The test contains 36 pairs across six English nouns: *bank*, *plant*,
*bat*, *crane*, *mouse*, and *spring*. The sentences were written with AI
assistance for this experiment. Each pair uses two different meanings of
the same word and counts as correct when both predictions are correct.

| Method | Sentences | Pairs | Nearby clue | Distracting clues | Distant clue |
|---|---:|---:|---:|---:|---:|
| Keyword rules | 10/72 | 3/36 | 1/12 | 1/12 | 1/12 |
| NLTK Lesk | 36/72 | 4/36 | 3/12 | 0/12 | 1/12 |
| MiniLM, full sentence | 41/72 | 9/36 | 4/12 | 0/12 | 5/12 |
| MiniLM, four words per side | 47/72 | 13/36 | 9/12 | 4/12 | 0/12 |

In `crane-late-01`, the crane opens its wings in one sentence and lifts
a steel beam in the other. Both actions occur beyond the crop. The
full-sentence model gets both senses right. The cropped inputs are
identical, so the windowed model makes the same prediction twice.

## SemEval-2015

The public benchmark contains 1,022 annotated occurrences. The XML
identifies the target and its part of speech. The evaluator considers
all WordNet candidates for that lemma and part of speech and accepts
any of the gold keys assigned to the occurrence.

| Method | Correct | Accuracy |
|---|---:|---:|
| WordNet first sense | 693/1,022 | 67.8% |
| MiniLM, full sentence | 509/1,022 | 49.8% |
| MiniLM, four words per side | 532/1,022 | 52.1% |

The first-sense rule was the strongest method in this run. Its score
matches the published reference. For comparison, the supervised
GlossBERT model trained on SemCor reported 80.4% F1 on this benchmark.
[Raganato et al., Table 2](https://aclanthology.org/E17-1010/)
· [Huang et al., Table 3](https://aclanthology.org/D19-1355/)

## Reproduce the results

From the repository root, run these commands in PowerShell with
Python 3.13 installed.

```powershell
py -3.13 -m venv .venv
$wsdPython = '.\.venv\Scripts\python.exe'
& $wsdPython -m pip install -r requirements.txt
& $wsdPython -m nltk.downloader wordnet
```

### Controlled context test

```powershell
foreach ($wsdMethod in "keywords", "lesk", "reranker", "reranker-window") {
    $wsdRows = & $wsdPython demo.py --method $wsdMethod --data data/context_pairs.csv
    Write-Host $wsdMethod
    $wsdRows | Select-Object -Last 2

    foreach ($wsdCondition in "near", "distractor", "late") {
        $wsdPattern = '-' + $wsdCondition + '-[0-9]+ True$'
        $wsdCorrect = @($wsdRows | Select-String -Pattern $wsdPattern).Count
        Write-Host "$wsdCondition $wsdCorrect/12"
    }
}
```

The first MiniLM run downloads the model. Inference runs on CPU.
[MiniLM model card](https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2)

### Public benchmark

Download the English evaluation data in Raganato format from the
versioned mirror. [Original SemEval task](https://aclanthology.org/S15-2049/)
· [Archive release](https://github.com/alvations/pywsd-datasets/releases/tag/v0.1.0)

```powershell
New-Item -ItemType Directory -Force data/external | Out-Null
Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/alvations/pywsd-datasets/releases/download/v0.1.0/raganato-v0.1.0.zip' -OutFile 'data/external/raganato-v0.1.0.zip'
Expand-Archive -LiteralPath 'data/external/raganato-v0.1.0.zip' -DestinationPath 'data/external' -Force
& $wsdPython evaluate_semeval.py
& $wsdPython evaluate_semeval.py --minilm
```