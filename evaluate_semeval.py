import argparse
import re
from pathlib import Path
from xml.etree import ElementTree as ET

from nltk.corpus import wordnet


DATA_DIR = (
    Path(__file__).parent
    / "data"
    / "external"
    / "WSD_Unified_Evaluation_Datasets"
    / "semeval2015"
)

WORDNET_POS = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}


def load_benchmark():
    corpus = ET.parse(DATA_DIR / "semeval2015.data.xml")
    gold = {}

    gold_path = DATA_DIR / "semeval2015.gold.key.txt"
    for line in gold_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields:
            instance_id, *keys = fields
            gold[instance_id] = set(keys)

    instance_ids = [node.attrib["id"] for node in corpus.iter("instance")]
    if len(instance_ids) != len(gold) or set(instance_ids) != set(gold):
        raise ValueError("The XML and gold files contain different instances.")

    return corpus, gold


def candidate_senses(instance):
    lemma = instance.attrib["lemma"]
    pos = WORDNET_POS[instance.attrib["pos"]]
    candidates = wordnet.lemmas(lemma, pos)
    if not candidates:
        raise ValueError(f"No WordNet senses for {lemma!r} ({pos}).")
    return candidates


def contexts_for(words, index):
    left = " ".join(words[:index])
    target = f"[{words[index]}]"
    right = " ".join(words[index + 1 :])
    full_context = " ".join((left, target, right)).strip()

    left_words = list(re.finditer(r"\b\w+\b", left))
    right_words = list(re.finditer(r"\b\w+\b", right))
    if len(left_words) >= 4:
        left = left[left_words[-4].start() :]
    if len(right_words) >= 4:
        right = right[: right_words[3].end()]

    window_context = " ".join((left, target, right)).strip()
    return full_context, window_context


def evaluate_minilm(corpus, gold):
    from reranker_baseline import choose_sense

    total = len(gold)
    processed = 0
    full_correct = 0
    window_correct = 0

    for sentence in corpus.iter("sentence"):
        tokens = list(sentence)
        words = [token.text or "" for token in tokens]

        for index, instance in enumerate(tokens):
            if instance.tag != "instance":
                continue

            definitions = {
                sense.key(): sense.synset().definition()
                for sense in candidate_senses(instance)
            }
            full_context, window_context = contexts_for(words, index)
            accepted_keys = gold[instance.attrib["id"]]

            if choose_sense(full_context, definitions) in accepted_keys:
                full_correct += 1
            if choose_sense(window_context, definitions) in accepted_keys:
                window_correct += 1

            processed += 1
            if processed % 100 == 0 or processed == total:
                print(f"Processed {processed}/{total}", flush=True)

    print(f"MiniLM full sentence: {full_correct}/{total} ({full_correct / total:.1%})")
    print(f"MiniLM four-word window: {window_correct}/{total} ({window_correct / total:.1%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minilm", action="store_true")
    args = parser.parse_args()

    corpus, gold = load_benchmark()
    if args.minilm:
        evaluate_minilm(corpus, gold)
        return

    correct = 0
    for instance in corpus.iter("instance"):
        instance_id = instance.attrib["id"]
        prediction = candidate_senses(instance)[0].key()
        if prediction in gold[instance_id]:
            correct += 1

    total = len(gold)
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {correct / total:.1%}")


if __name__ == "__main__":
    main()