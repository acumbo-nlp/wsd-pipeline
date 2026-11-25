import argparse
import csv
from pathlib import Path

from sense_inventory import SENSES_BY_TARGET


KEYWORDS_BY_TARGET = {
    "bank": {"river": "river bank", "money": "financial institution"},
    "plant": {"workers": "factory", "leaves": "living plant"},
    "bat": {"wings": "flying mammal", "ball": "sports bat"},
    "crane": {"boom": "lifting machine", "feathers": "bird"},
    "mouse": {"cheese": "rodent", "cursor": "computer mouse"},
    "spring": {"blossoms": "season", "coil": "mechanical spring"},
}


def predict_with_keywords(sentence, target):
    for keyword, label in KEYWORDS_BY_TARGET.get(target, {}).items():
        if keyword in sentence:
            return label
    return "unknown"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--method",
    choices=["keywords", "lesk", "reranker", "reranker-window"],
    default="keywords",
)
parser.add_argument("--data", default="data/dev_pairs.csv")
args = parser.parse_args()

predict_sense = predict_with_keywords
if args.method == "lesk":
    from lesk_baseline import predict_with_lesk

    predict_sense = predict_with_lesk
elif args.method == "reranker":
    from reranker_baseline import predict_with_reranker

    predict_sense = predict_with_reranker
elif args.method == "reranker-window":
    from reranker_baseline import predict_with_window

    predict_sense = predict_with_window



csv_path = Path(__file__).parent / args.data
checks_by_pair = {}

with csv_path.open(encoding="utf-8", newline="") as csv_file:
    for row in csv.DictReader(csv_file):
        pair_id = row["pair_id"]
        target = row["target"].strip().lower()
        if target not in SENSES_BY_TARGET:
            raise ValueError(f"Unsupported target: {target!r}")
        prediction = predict_sense(row["sentence"], target)
        is_correct = prediction == row["expected_sense"]

        if pair_id not in checks_by_pair:
            checks_by_pair[pair_id] = []
        checks_by_pair[pair_id].append(is_correct)

total_sentences = 0
correct_sentences = 0
correct_pairs = 0

for pair_id, checks in checks_by_pair.items():
    if len(checks) != 2:
        raise ValueError(f"Pair {pair_id} must contain exactly two sentences.")

    pair_correct = all(checks)
    total_sentences += len(checks)
    correct_sentences += checks.count(True)

    if pair_correct:
        correct_pairs += 1

    print(pair_id, pair_correct)

print(f"Sentences correct: {correct_sentences}/{total_sentences}")
print(f"Pairs correct: {correct_pairs}/{len(checks_by_pair)}")